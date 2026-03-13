#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "esp_wifi.h"
#include "esp_timer.h"
#include <math.h>

#define ENABLE_HR_SENSOR 1

const char *ssid = "txpulse";
const uint16_t UDP_PORT = 4210;

MAX30105 sensor;
WiFiUDP udp;
SemaphoreHandle_t serialMutex;

const byte RATE_SIZE = 4;
byte rates[RATE_SIZE];
byte rateSpot = 0;
byte rateCount = 0;
long lastBeatMs = 0;
int beatAvg = -1;
bool beatValid = false;
uint64_t lastBeatUpdateUs = 0;
volatile uint32_t csiSeq = 0;
uint64_t lastBpmLineUs = 0;
uint64_t lastStatUs = 0;
uint64_t lastStatusUs = 0;
volatile uint32_t csiCountStat = 0;
uint32_t txPktSeen = 0;
bool hrSensorReady = false;
const bool PRINT_RX_PKT = false;
const bool PRINT_STAT = false;

struct CsiRecord {
  uint64_t ts;
  uint32_t seq;
  int rssi;
  int len;
  float amps[64];
};

static QueueHandle_t csiQueue = nullptr;
static const int CSI_QUEUE_LEN = 160;

void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;

  int offset = info->first_word_invalid ? 4 : 0;
  int usableBytes = info->len - offset;
  int iqPairs = usableBytes / 2;
  if (iqPairs <= 0) return;

  CsiRecord rec;
  rec.ts = (uint64_t)esp_timer_get_time();
  rec.seq = ++csiSeq;
  csiCountStat++;
  rec.rssi = (int)info->rx_ctrl.rssi;
  rec.len = (int)info->len;

  for (int sc = 0; sc < 64; sc++) {
    if (sc < iqPairs) {
      int idx = offset + sc * 2;
      int8_t imag = (int8_t)info->buf[idx];
      int8_t real = (int8_t)info->buf[idx + 1];
      rec.amps[sc] = sqrtf((float)(real * real + imag * imag));
    } else {
      rec.amps[sc] = 0.0f;
    }
  }

  xQueueSend(csiQueue, &rec, 0);
}

void CsiPrintTask(void *pvParameters) {
  CsiRecord rec;
  char line[640];

  for (;;) {
    if (xQueueReceive(csiQueue, &rec, portMAX_DELAY) != pdTRUE) continue;

    int pos = snprintf(line, sizeof(line), "CSI_PKT,%llu,%lu,%d,%d",
                       (unsigned long long)rec.ts,
                       (unsigned long)rec.seq,
                       rec.rssi, rec.len);

    for (int sc = 0; sc < 64; sc++) {
      int whole = (int)rec.amps[sc];
      int frac = (int)((rec.amps[sc] - (float)whole) * 1000.0f);
      if (frac < 0) frac = -frac;
      pos += snprintf(line + pos, sizeof(line) - pos, ",%d.%03d", whole, frac);
    }
    line[pos++] = '\n';

    xSemaphoreTake(serialMutex, portMAX_DELAY);
    Serial.write((const uint8_t *)line, pos);
    xSemaphoreGive(serialMutex);
  }
}

void HeartRateTask(void *pvParameters) {
  for (;;) {
    if (!hrSensorReady) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    long ir = sensor.getIR();

    if (ir > 20000 && checkForBeat(ir)) {
      long now = millis();
      long delta = now - lastBeatMs;
      lastBeatMs = now;

      if (delta > 0) {
        float bpm = 60.0f / (delta / 1000.0f);

        if (bpm > 35 && bpm < 220) {
          rates[rateSpot++] = (byte)lroundf(bpm);
          rateSpot %= RATE_SIZE;
          if (rateCount < RATE_SIZE) rateCount++;

          int sum = 0;
          for (byte i = 0; i < rateCount; i++) sum += rates[i];
          beatAvg = sum / rateCount;
          beatValid = true;
          lastBeatUpdateUs = (uint64_t)esp_timer_get_time();
        }
      }
    }
    if (beatValid) {
      uint64_t nowUs = (uint64_t)esp_timer_get_time();
      if (nowUs - lastBeatUpdateUs > 2000000ULL) {
        beatValid = false;
      }
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }
}

void setup() {
  Serial.begin(921600);
  delay(300);

  serialMutex = xSemaphoreCreateMutex();
  csiQueue = xQueueCreate(CSI_QUEUE_LEN, sizeof(CsiRecord));

  if (ENABLE_HR_SENSOR) {
    Wire.begin();
    if (sensor.begin(Wire, I2C_SPEED_FAST)) {
      sensor.setup();
      sensor.setPulseAmplitudeRed(0x1F);
      sensor.setPulseAmplitudeGreen(0);
      hrSensorReady = true;
      Serial.println("HR_SENSOR_READY");
    } else {
      hrSensorReady = false;
      Serial.println("HR_SENSOR_NOT_FOUND_CONTINUING_WITH_CSI_ONLY");
    }
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid);
  while (WiFi.status() != WL_CONNECTED) delay(200);

  udp.begin(UDP_PORT);

  wifi_csi_config_t csi_cfg = {};
  csi_cfg.lltf_en = true;
  csi_cfg.htltf_en = true;
  csi_cfg.stbc_htltf2_en = true;
  csi_cfg.ltf_merge_en = true;
  csi_cfg.channel_filter_en = false;
  csi_cfg.manu_scale = false;
  csi_cfg.shift = 0;

  esp_wifi_set_promiscuous(true);
  esp_wifi_set_csi_rx_cb(&wifi_csi_rx_cb, nullptr);
  esp_wifi_set_csi_config(&csi_cfg);
  esp_wifi_set_csi(true);

  xTaskCreatePinnedToCore(HeartRateTask, "HR_Task", 4096, nullptr, 3, nullptr, 1);
  xTaskCreatePinnedToCore(CsiPrintTask, "CSI_Print", 8192, nullptr, 1, nullptr, 1);

  Serial.println("RX_READY");
  uint8_t ch;
  wifi_second_chan_t ch2;
  esp_wifi_get_channel(&ch, &ch2);
  Serial.print("RX_STATUS,wifi=1,ssid=");
  Serial.print(ssid);
  Serial.print(",rssi=");
  Serial.print(WiFi.RSSI());
  Serial.print(",ip=");
  Serial.print(WiFi.localIP());
  Serial.print(",channel=");
  Serial.println((int)ch);
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char buf[96];
    int n = udp.read(buf, sizeof(buf) - 1);
    if (n > 0) {
      buf[n] = '\0';
      txPktSeen++;
      if (PRINT_RX_PKT) {
        char rxLine[128];
        int rlen = snprintf(rxLine, sizeof(rxLine), "RX_PKT,%s\n", buf);
        if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
          Serial.write((const uint8_t *)rxLine, rlen);
          xSemaphoreGive(serialMutex);
        }
      }
    }
  }

  uint64_t nowUs = (uint64_t)esp_timer_get_time();

  if (nowUs - lastBpmLineUs >= 100000ULL) {
    lastBpmLineUs = nowUs;
    int bpmOut = beatAvg;
    int validOut = beatValid ? 1 : 0;
    long sensorAgeMs = validOut ? (long)((nowUs - lastBeatUpdateUs) / 1000ULL) : -1;

    char bpmLine[80];
    int len = snprintf(bpmLine, sizeof(bpmLine), "BPM,%llu,%d,%d,%ld\n",
                       (unsigned long long)nowUs, bpmOut, validOut, sensorAgeMs);

    if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
      Serial.write((const uint8_t *)bpmLine, len);
      xSemaphoreGive(serialMutex);
    }
  }

  if (nowUs - lastStatusUs >= 1000000ULL) {
    lastStatusUs = nowUs;
    int wifiConnected = (WiFi.status() == WL_CONNECTED) ? 1 : 0;
    int rssi = wifiConnected ? WiFi.RSSI() : 0;
    uint32_t pps = csiCountStat;
    csiCountStat = 0;

    char statusLine[128];
    int len = snprintf(statusLine, sizeof(statusLine),
                       "STATUS,wifi=%d,rssi=%d,tx_pkt=%lu,csi_pps=%lu\n",
                       wifiConnected, rssi,
                       (unsigned long)txPktSeen, (unsigned long)pps);

    if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
      Serial.write((const uint8_t *)statusLine, len);
      xSemaphoreGive(serialMutex);
    }
  }

  if (PRINT_STAT && nowUs - lastStatUs >= 1000000ULL) {
    lastStatUs = nowUs;
    char statLine[128];
    int slen = snprintf(statLine, sizeof(statLine),
                        "STAT,ts_us=%llu,csi_pps=%lu,tx_pkt_seen=%lu,bpm_valid=%d\n",
                        (unsigned long long)nowUs, (unsigned long)csiCountStat,
                        (unsigned long)txPktSeen, beatValid ? 1 : 0);
    if (xSemaphoreTake(serialMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
      Serial.write((const uint8_t *)statLine, slen);
      xSemaphoreGive(serialMutex);
    }
  }

  vTaskDelay(pdMS_TO_TICKS(2));
}
