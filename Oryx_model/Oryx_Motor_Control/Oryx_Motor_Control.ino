int relayR1 = 2;
int relayR2 = 3;

int relayL1 = 4;
int relayL2 = 5;

void setup() {
  pinMode(relayR1, OUTPUT);
  pinMode(relayR2, OUTPUT);
  pinMode(relayL1, OUTPUT);
  pinMode(relayL2, OUTPUT);

  digitalWrite(relayR1, HIGH);
  digitalWrite(relayR2, HIGH);
  digitalWrite(relayL1, HIGH);
  digitalWrite(relayL2, HIGH);

  Serial.begin(9600);
  Serial.println("Relay Motor Control Ready");
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();

    if (command == '1') {
      Serial.println("Forward");
      digitalWrite(relayR1, LOW);  
      digitalWrite(relayR2, HIGH);
      digitalWrite(relayL1, LOW);  
      digitalWrite(relayL2, HIGH);
    } 
    else if (command == '2') {
      Serial.println("Backward");
      digitalWrite(relayR1, HIGH); 
      digitalWrite(relayR2, LOW);
      digitalWrite(relayL1, HIGH); 
      digitalWrite(relayL2, LOW);
    } 
    else if (command == '0') {
      Serial.println("Stop");
      digitalWrite(relayR1, HIGH);
      digitalWrite(relayR2, HIGH);
      digitalWrite(relayL1, HIGH);
      digitalWrite(relayL2, HIGH);
    }
  }
}
