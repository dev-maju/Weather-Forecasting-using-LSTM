#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Firebase_ESP_Client.h>
#include <addons/RTDBHelper.h>

// ====================
// Rain sensor
// ====================
#define RAIN_ANALOG_PIN 34  

// Ultrasonic sensor pins
#define ULTRASONIC_TRIG 5
#define ULTRASONIC_ECHO 18

// ====================
// Wi-Fi credentials
// ====================
#define WIFI_SSID "iPhone"
#define WIFI_PASSWORD "12345678"

// ====================
// Firebase credentials
// ====================
#define API_KEY "AIzaSyC0tH-5g1wUJTR5X_3Jg3KzKfWimJRVCYo"
#define DATABASE_URL "https://test1-a7529-default-rtdb.firebaseio.com/"

// ====================
// Prediction server
// ====================
#define PREDICTION_SERVER "http://172.20.10.2:8000/sensor"

// ====================
// Objects
// ====================
Adafruit_BME280 bme;  
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

unsigned long previousMillis = 0;
const long interval = 2000; // keep small for testing (change to 3600000 for 1 hour)

// ========================================================
// Function: Send sensor data to prediction server
// ========================================================
void sendToPredictionServer(float temp, float hum, float pres)
{
  HTTPClient http;

  http.begin(PREDICTION_SERVER);
  http.addHeader("Content-Type", "application/json");

  String payload = "{";
  payload += "\"temperature\":" + String(temp) + ",";
  payload += "\"humidity\":" + String(hum) + ",";
  payload += "\"pressure\":" + String(pres);
  payload += "}";

  int httpResponseCode = http.POST(payload);

  if(httpResponseCode > 0)
  {
    String response = http.getString();
    Serial.println("🔮 Prediction server response:");
    Serial.println(response);
  }
  else
  {
    Serial.print("❌ Prediction server error: ");
    Serial.println(httpResponseCode);
  }

  http.end();
}

void setup() {

  Serial.begin(115200);
  delay(1000);

  // Connect WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.println("\n✅ WiFi connected");

  // Initialize BME280
  if (!bme.begin(0x77)) {
    Serial.println("❌ Could not find BME280 sensor");
    while (1);
  }

  Serial.println("✅ BME280 sensor found");

  // Firebase config
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  if (!Firebase.signUp(&config, &auth, "", "")) {
    Serial.printf("❌ Firebase signUp failed: %s\n",
                  config.signer.signupError.message.c_str());
  } else {
    Serial.println("✅ Firebase anonymous signUp successful");
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  Serial.println("Firebase initialized");

  pinMode(RAIN_ANALOG_PIN, INPUT);

  pinMode(ULTRASONIC_TRIG, OUTPUT);
  pinMode(ULTRASONIC_ECHO, INPUT);
}

void loop() {

  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {

    previousMillis = currentMillis;

    // ====================
    // Read BME280
    // ====================
    float temperature = bme.readTemperature();
    float humidity = bme.readHumidity();
    float pressure = bme.readPressure() / 100.0F;

    Serial.printf("🌡 Temp: %.2f °C, 💧 Hum: %.2f %%, ⬇ Pressure: %.2f hPa\n",
                  temperature, humidity, pressure);

    // Send data to prediction server
    sendToPredictionServer(temperature, humidity, pressure);

    // ====================
    // Upload to Firebase
    // ====================
    Firebase.RTDB.setFloat(&fbdo, "/ESP32/temperature", temperature);
    Firebase.RTDB.setFloat(&fbdo, "/ESP32/humidity", humidity);
    Firebase.RTDB.setFloat(&fbdo, "/ESP32/pressure", pressure);

    // ====================
    // Rain sensor
    // ====================
    int rainAnalogValue = analogRead(RAIN_ANALOG_PIN);

    float rainPercent =
      (1.0f - (float)rainAnalogValue / 4095.0f) * 100.0f;

    rainPercent = constrain(rainPercent, 0, 100);

    Serial.printf("🌧 Rain Sensor: %d (%.2f%% wetness)\n",
                  rainAnalogValue, rainPercent);

    Firebase.RTDB.setFloat(&fbdo,
                           "/ESP32/rain_percent",
                           rainPercent);

    // ====================
    // Ultrasonic sensor
    // ====================
    digitalWrite(ULTRASONIC_TRIG, LOW);
    delayMicroseconds(2);

    digitalWrite(ULTRASONIC_TRIG, HIGH);
    delayMicroseconds(10);

    digitalWrite(ULTRASONIC_TRIG, LOW);

    long duration = pulseIn(ULTRASONIC_ECHO, HIGH);

    float distance = (duration * 0.0343) / 2.0;

    Serial.printf("📏 Water Distance: %.2f cm\n", distance);

    Firebase.RTDB.setFloat(&fbdo,
                           "/ESP32/water_distance",
                           distance);

    Serial.println("--------------------------------");
  }
}