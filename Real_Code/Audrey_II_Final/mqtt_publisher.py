import paho.mqtt.client as mqtt

class MqttPublisher: 
    def __init__(self, broker_address, broker_port=1883, client_id="mqtt_client" ):

        #Initializes the MQTT publisher
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.client_id = client_id
        self.client = mqtt.Client(client_id)

    def connect(self):

        #Connects to the MQTT broker
        try:
            self.client.connect(self.broker_address, self.broker_port)
            print("Connected to MQTT broker!")
        except Exception as e: 
            print()(f"Failed to connect to MQTT broker: {e}")

    def publish(self, topic, message):

        #Published a message to an MQTT topic
        try: 
            self.client.publish(topic, message)
            print(f"Sent '{message}'")
        except Exception as e: 
            print(f"Failed to publish message: {e}")

    def disconnect(self):

        #Disconnects from the MQTT broker
        try:
            self.client.disconnect()
            print("Disconnected from MQTT broker")
        except Exception as e:
            print(f"Failed to disconnect from broker: {e}")

        


