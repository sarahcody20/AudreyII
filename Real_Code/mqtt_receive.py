import paho.mqtt.client as mqtt

# Define the broker address and port
broker_address = "10.5.10.72"  # Replace with your broker's address
broker_port = 1885              # Replace with your broker's port

# Define a flag to indicate when a message is received
message_received = False

# Callback function that gets called when a message is received
def on_message(client, userdata, message):
    global message_received
    print(f"Received message: {message.payload.decode()} on topic {message.topic}")
    message_received = True  # Set the flag to True when a message is received

# Create a new MQTT client instance
client = mqtt.Client("subscriber_client")
client.on_message = on_message  # Assign the callback function

# Connect to the MQTT broker
client.connect(broker_address, broker_port)

# Subscribe to the "test" topic
client.subscribe("test")

# Start the loop to process callbacks
client.loop_start()

print("Waiting for messages...")

# Wait until a message is received
try:
    while not message_received:
        pass  # Just wait until a message is received

finally:
    # Stop the loop and disconnect from the broker
    client.loop_stop()
    client.disconnect()
    print("Disconnected from the broker.")
