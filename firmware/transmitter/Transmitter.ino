// ================= TX (TransmitterESP32.ino) =================
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <esp_timer.h>

const char *ssid = "txpulse";
const uint16_t UDP_PORT = 4210;
// Use unicast to avoid AP DTIM buffering of broadcast/multicast frames
// which can collapse effective packet delivery rate on the receiver.
const IPAddress receiverIP(192, 168, 4, 2);
const uint32_t TX_PERIOD_MS = 12; // ~83 Hz (close to 80 Hz target)

WiFiUDP udp;
uint32_t seq = 0;
uint64_t lastStatusUs = 0;

void setup() {
  Serial.begin(115200);
  delay(300);

  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(ssid, nullptr, 1, 0, 1); // channel 1
  Serial.print("AP,");
  Serial.println(ok ? "OK" : "FAIL");

  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  udp.begin(UDP_PORT);

  Serial.print("AP_IP,");
  Serial.println(WiFi.softAPIP());
  Serial.println("TX_READY");
}

void loop() {
  char payload[48];
  uint64_t txTsUs = (uint64_t)esp_timer_get_time();
  snprintf(payload, sizeof(payload), "S,%lu,%llu",
           (unsigned long)seq,
           (unsigned long long)txTsUs);

  udp.beginPacket(receiverIP, UDP_PORT);
  udp.write((const uint8_t *)payload, strlen(payload));
  udp.endPacket();
  seq++;

  if (txTsUs - lastStatusUs >= 1000000ULL) {
    lastStatusUs = txTsUs;
    Serial.print("TX_STAT,seq=");
    Serial.println((unsigned long)seq);
  }

  delay(TX_PERIOD_MS);
}
