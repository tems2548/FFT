#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_adc/adc_continuous.h"
#include "driver/uart.h"

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"

#define TIMES 256
#define SAMPLE_FREQ 8000 // 4 kHz sampling rate
#define READ_LEN 512      // Bytes to read per block
#define EXAMPLE_ADC_UNIT ADC_UNIT_1
#define EXAMPLE_ADC_CHAN ADC_CHANNEL_6 // ESP32-S3: GPIO7

// Streaming to FFT.py over the same USB-UART used for logging/flashing.
// Raw ADC samples are decimated (averaged) before being sent, since 80 kHz
// of raw samples can't fit over UART at the default 115200 baud. Packets
// are framed with an ASCII magic word so the Python side can resync even
// if ESP_LOG lines land in the stream.
//
// The ADC continuous driver's actual conversion rate doesn't reliably
// match the sample_freq_hz configured on it (observed both ~2x over and
// ~2x under depending on the requested rate), so instead of a fixed
// decimation factor, the firmware measures the real raw rate at startup
// (see measure_actual_raw_sample_rate) and derives the decimation factor
// needed to land close to TARGET_OUTPUT_RATE_HZ from that measurement.
#define TARGET_OUTPUT_RATE_HZ 4000
#define PACKET_SAMPLES 128
#define STREAM_UART_PORT UART_NUM_0

static const char *TAG = "ADC_DMA";
static TaskHandle_t s_processing_task_handle = NULL;

// NEW: Global handle for the calibration scheme
static adc_cali_handle_t s_cali_handle = NULL;
static bool s_calibrated = false;

// ESP-IDF's default console output writes to UART0 via a polling ROM path
// that never installs an interrupt-driven driver, but uart_write_bytes()
// requires one. Install it once (skip if something else, e.g. esp_console,
// already did).
static void init_uart_stream(void)
{
    if (!uart_is_driver_installed(STREAM_UART_PORT)) {
        ESP_ERROR_CHECK(uart_driver_install(STREAM_UART_PORT, 2048, 0, 0, NULL, 0));
    }
}

// Sends a one-shot header so Python learns the true decimated sample rate
// instead of having to hardcode it.
static void send_meta_packet(uint32_t output_sample_rate)
{
    uint8_t buf[8];
    memcpy(buf, "META", 4);
    memcpy(buf + 4, &output_sample_rate, sizeof(output_sample_rate));
    uart_write_bytes(STREAM_UART_PORT, (const char *)buf, sizeof(buf));
}

// The ADC continuous driver's actual conversion rate doesn't reliably match
// the sample_freq_hz passed to adc_continuous_config() (observed ~2x off on
// ESP32-S3). Rather than trust the configured value, measure real raw
// samples/sec for ~1s at startup and derive the decimated output rate from
// that, so the frequency axis Python builds from the META packet is
// accurate regardless of the actual hardware rate.
static uint32_t measure_actual_raw_sample_rate(adc_continuous_handle_t handle)
{
    uint8_t result_buffer[READ_LEN];
    uint32_t ret_num = 0;
    uint64_t valid_count = 0;
    const int64_t measure_us = 1000000;   // measurement window, once data is actually flowing
    const int64_t max_wait_us = 3000000;  // give up if no sample arrives at all within this
    int64_t entry_time = esp_timer_get_time();
    int64_t window_start = -1;            // set on the first real sample, not on entry

    while (1)
    {
        int64_t now = esp_timer_get_time();
        if (window_start >= 0 && now - window_start >= measure_us) break;
        if (window_start < 0 && now - entry_time >= max_wait_us) break;

        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(50));
        while (1)
        {
            esp_err_t ret = adc_continuous_read(handle, result_buffer, READ_LEN, &ret_num, 0);
            if (ret != ESP_OK) break;
            for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES)
            {
                adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result_buffer[i];
                if (p->type2.unit == EXAMPLE_ADC_UNIT && p->type2.channel == EXAMPLE_ADC_CHAN)
                {
                    if (window_start < 0) window_start = esp_timer_get_time();
                    valid_count++;
                }
            }
        }
    }

    if (window_start < 0 || valid_count == 0)
    {
        ESP_LOGE(TAG, "No ADC samples arrived during rate measurement; is the ADC actually running?");
        return 1; // avoid propagating 0 into a divide-by-zero downstream
    }

    int64_t elapsed_us = esp_timer_get_time() - window_start;
    if (elapsed_us <= 0) elapsed_us = 1;
    uint32_t raw_rate = (uint32_t)((valid_count * 1000000ULL) / (uint64_t)elapsed_us);
    ESP_LOGI(TAG, "Measured raw ADC rate: %" PRIu32 " Hz (configured sample_freq_hz was %d)", raw_rate, SAMPLE_FREQ);
    return raw_rate;
}

// samples[] are millivolts, little-endian int16, matching the ESP32's
// native (little-endian) byte order so Python can unpack with "<h".
static void send_data_packet(const int16_t *samples, uint16_t count)
{
    uint8_t header[6];
    memcpy(header, "DATA", 4);
    memcpy(header + 4, &count, sizeof(count));
    uart_write_bytes(STREAM_UART_PORT, (const char *)header, sizeof(header));
    uart_write_bytes(STREAM_UART_PORT, (const char *)samples, count * sizeof(int16_t));
}

// ISR Callback triggered when the DMA buffer fills up
static bool IRAM_ATTR adc_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data)
{
    BaseType_t mustYield = pdFALSE;
    if (s_processing_task_handle != NULL) {
        vTaskNotifyGiveFromISR(s_processing_task_handle, &mustYield);
    }
    return (mustYield == pdTRUE);
}

// Data Processing Task
static void adc_continuous_processing_task(void *pvParameters)
{
    adc_continuous_handle_t handle = (adc_continuous_handle_t)pvParameters;
    uint8_t result_buffer[READ_LEN] = {0};
    uint32_t ret_num = 0;
    
    uint32_t sample_count = 0;
    uint64_t sample_sum = 0;

    // NEW: Variables to track the peaks of the sine wave
    uint32_t max_raw = 0;
    uint32_t min_raw = 4095; // Start at max possible value so it drops immediately

    int64_t last_print_time = esp_timer_get_time();
    int64_t last_meta_time = esp_timer_get_time();

    // Decimation + streaming state
    uint64_t decim_sum = 0;
    uint32_t decim_count = 0;
    int16_t packet_buf[PACKET_SAMPLES];
    uint16_t packet_idx = 0;

    uint32_t measured_raw_rate = measure_actual_raw_sample_rate(handle);
    uint32_t decimation = measured_raw_rate / TARGET_OUTPUT_RATE_HZ;
    if (decimation < 1) decimation = 1;
    uint32_t output_sample_rate = measured_raw_rate / decimation;
    ESP_LOGI(TAG, "Decimating by %" PRIu32 " -> streaming at %" PRIu32 " Hz (target was %d Hz)",
             decimation, output_sample_rate, TARGET_OUTPUT_RATE_HZ);
    send_meta_packet(output_sample_rate);

    while (1)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        while (1)
        {
            esp_err_t ret = adc_continuous_read(handle, result_buffer, READ_LEN, &ret_num, 0);

            if (ret == ESP_OK)
            {
                for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES)
                {
                    adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result_buffer[i];
                    uint32_t chan = p->type2.channel;
                    uint32_t data = p->type2.data;
                    uint32_t unit = p->type2.unit;

                    if (unit == EXAMPLE_ADC_UNIT && chan == EXAMPLE_ADC_CHAN)
                    {
                        // NEW: Capture the Peaks and Troughs instantly
                        if (data > max_raw) max_raw = data;
                        if (data < min_raw) min_raw = data;

                        sample_sum += data;
                        sample_count++;

                        // NEW: decimate raw samples and stream them to FFT.py
                        decim_sum += data;
                        decim_count++;
                        if (decim_count >= decimation)
                        {
                            uint32_t decim_avg_raw = (uint32_t)(decim_sum / decim_count);
                            decim_sum = 0;
                            decim_count = 0;

                            if (s_calibrated)
                            {
                                int v_mv = 0;
                                adc_cali_raw_to_voltage(s_cali_handle, decim_avg_raw, &v_mv);
                                packet_buf[packet_idx++] = (int16_t)v_mv;
                                if (packet_idx >= PACKET_SAMPLES)
                                {
                                    send_data_packet(packet_buf, packet_idx);
                                    packet_idx = 0;
                                }
                            }
                        }
                    }
                }

                int64_t current_time = esp_timer_get_time();

                // Resend META every second. It's only sent once at boot
                // otherwise, so a Python client that connects after the
                // board has already been running (no reset on port-open,
                // or a reconnect) would wait forever for a rate it missed.
                if ((current_time - last_meta_time) >= 1000000)
                {
                    send_meta_packet(output_sample_rate);
                    last_meta_time = current_time;
                }

                // NEW: Update 10 times a second (100,000 microseconds) for a more responsive reading
                if ((current_time - last_print_time) >= 100000)
                {
                    if (sample_count > 0) {
                        uint32_t average_raw = (uint32_t)(sample_sum / sample_count);
                        
                        int v_avg_mv = 0, v_max_mv = 0, v_min_mv = 0;
                        
                        if (s_calibrated) {
                            // Convert your key metrics to real voltage
                            adc_cali_raw_to_voltage(s_cali_handle, average_raw, &v_avg_mv);
                            adc_cali_raw_to_voltage(s_cali_handle, max_raw, &v_max_mv);
                            adc_cali_raw_to_voltage(s_cali_handle, min_raw, &v_min_mv);
                            
                            // Calculate Peak-to-Peak Amplitude
                            int v_ptp_mv = v_max_mv - v_min_mv;

                            ESP_LOGI(TAG, "Sine Wave | DC Bias: %d mV | V_Max: %d mV | V_Min: %d mV | Peak-to-Peak: %d mV", 
                                     v_avg_mv, v_max_mv, v_min_mv, v_ptp_mv);
                        }
                    }
                    // Reset accumulators for the next batch
                    sample_count = 0;
                    sample_sum = 0;
                    max_raw = 0;
                    min_raw = 4095;
                    last_print_time = current_time;
                }
            }
            else if (ret == ESP_ERR_TIMEOUT) { break; }
            else if (ret == ESP_ERR_INVALID_STATE)
            {
                adc_continuous_stop(handle);
                adc_continuous_start(handle);
                break;
            }
        }
    }
}
// NEW: Calibration Initialization Function
static void init_adc_calibration(void)
{
    esp_err_t ret = ESP_FAIL;
    
    // ESP32-S3 uses the Curve Fitting calibration scheme
#if ADC_CALI_SCHEME_CURVE_FITTING_SUPPORTED
    adc_cali_curve_fitting_config_t cali_config = {
        .unit_id = EXAMPLE_ADC_UNIT,
        .chan = EXAMPLE_ADC_CHAN,
        .atten = ADC_ATTEN_DB_12,
        .bitwidth = SOC_ADC_DIGI_MAX_BITWIDTH,
    };
    ret = adc_cali_create_scheme_curve_fitting(&cali_config, &s_cali_handle);
#endif

    if (ret == ESP_OK) {
        s_calibrated = true;
        ESP_LOGI(TAG, "ADC Calibration initialized successfully.");
    } else {
        ESP_LOGE(TAG, "ADC Calibration failed or unsupported on this chip.");
    }
}

void init_adc_continuous(adc_continuous_handle_t *out_handle)
{
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096, 
        .conv_frame_size = READ_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, out_handle));

    adc_continuous_config_t config = {
        .pattern_num = 1,
        .sample_freq_hz = SAMPLE_FREQ,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1, 
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2, 
    };

    adc_digi_pattern_config_t adc_pattern = {
        .atten = ADC_ATTEN_DB_12, 
        .channel = EXAMPLE_ADC_CHAN,
        .unit = EXAMPLE_ADC_UNIT,
        .bit_width = SOC_ADC_DIGI_MAX_BITWIDTH,
    };
    config.adc_pattern = &adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(*out_handle, &config));

    adc_continuous_evt_cbs_t cbs = {
        .on_conv_done = adc_conv_done_cb,
    };
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(*out_handle, &cbs, NULL));
}

void app_main(void)
{
    adc_continuous_handle_t handle = NULL;

    // 1. Initialize Calibration First
    init_adc_calibration();

    // 2. Initialize the UART driver used to stream samples to FFT.py
    init_uart_stream();

    // 3. Initialize Hardware
    init_adc_continuous(&handle);

    // 4. Start Hardware *before* creating the processing task. The task
    //    runs at higher priority than app_main and begins its startup rate
    //    measurement (see measure_actual_raw_sample_rate) the instant it's
    //    created, so starting the ADC first ensures that window isn't
    //    ticking down before there's any data to count.
    ESP_ERROR_CHECK(adc_continuous_start(handle));
    ESP_LOGI(TAG, "ADC Continuous mode initialized and started.");

    // 5. Create Task
    TaskHandle_t tmp_handle = NULL;
    xTaskCreate(adc_continuous_processing_task, "adc_process", 4096, handle, 5, &tmp_handle);
    s_processing_task_handle = tmp_handle;

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}