import paho.mqtt.client as mqtt  # Correctly import the client module

def mqtt_connection():
    broker_address = "10.5.10.72"
    broker_port = 1885

    x = 10
    
    # Create a new MQTT client instance with the correct callback API version
    client = mqtt.Client("gripper")  # protocol version can also be adjusted

    # Connect to the broker
    client.connect(broker_address, broker_port)
    
    # Publish a message
    client.publish("test", x)
    print("message sent")
    
    # Disconnect from the broker
    client.disconnect()

# Call the function

mqtt_connection()

