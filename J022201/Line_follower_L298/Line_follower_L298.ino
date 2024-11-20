int IN1 = 2 ;
int IN2 = 4 ;
int IN3 = 7 ;
int IN4 = 8 ;
int ENA = 3;
int ENB = 5;
int Lsensor = A1;
int Rsensor = A0;
int RIGHT , LEFT;
int x = 70;
int y = 80;


void setup() {
  
Serial.begin(9600);
pinMode(IN1 , OUTPUT);
pinMode(IN2 , OUTPUT);
pinMode(IN3 , OUTPUT);
pinMode(IN4 , OUTPUT);
pinMode(ENA , OUTPUT);
pinMode(ENB , OUTPUT);
pinMode(Rsensor , INPUT);
pinMode(Lsensor , INPUT);

}


void loop() {

  RIGHT = digitalRead(Rsensor);
  LEFT = digitalRead(Lsensor);

  if (RIGHT == 0 && LEFT == 0) {
  
    digitalWrite(IN1 , HIGH);
    digitalWrite(IN2 , LOW);
    digitalWrite(IN3 , HIGH);
    digitalWrite(IN4 , LOW);
    analogWrite(ENA , x);
    analogWrite(ENB , y);
    Serial.println("FORWARD");
  
  }

  else if (RIGHT == 0 && LEFT == 1) {
  
    digitalWrite(IN1 , HIGH);
    digitalWrite(IN2 , LOW);
    digitalWrite(IN3 , LOW);
    digitalWrite(IN4 , HIGH);
    analogWrite(ENA , x);
    analogWrite(ENB , y);
    Serial.println("LEFT");
  
  }

   else if (RIGHT == 1 && LEFT == 0) {
  
    digitalWrite(IN1 , LOW);
    digitalWrite(IN2 , HIGH);
    digitalWrite(IN3 , HIGH);
    digitalWrite(IN4 , LOW);
    analogWrite(ENA , x);
    analogWrite(ENB , y);
    Serial.println("RIGHT");
  
  }
}
