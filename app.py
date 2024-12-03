import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import mysql.connector
from opencage.geocoder import OpenCageGeocode
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from pathlib import Path
from dotenv import load_dotenv
import os
from twilio.rest import Client
from sklearn.model_selection import train_test_split
from sklearnex.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import openpyxl
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Threading.*")

# Initialize database connection
load_dotenv()

# Get the database credentials from the .env file
mydb = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME")
)
cursor = mydb.cursor()

# OpenCage Geocoder API key
opencage_api_key = os.getenv("opencage_api_key")
geocoder = OpenCageGeocode(opencage_api_key)

# Function to fetch producer and consumer locations
def fetch_producer_consumer_locations():
    cursor.execute("SELECT latitude, longitude FROM produce")
    producers = cursor.fetchall()

    cursor.execute("SELECT latitude, longitude FROM consumers")
    consumers = cursor.fetchall()

    return producers, consumers
data = pd.read_csv('crop_data.csv')
# Function to get latitude and longitude from a location
def get_lat_long(location):
    results = geocoder.geocode(location)
    if results and len(results):
        latitude = results[0]['geometry']['lat']
        longitude = results[0]['geometry']['lng']
        return latitude, longitude
    else:
        return None, None

area_files = {
    'ECIL': 'ECIL.xlsx',
    'ABIDS': 'ABIDS.xlsx',
    'MEHDIPATNAM': 'MEHEDHIPATNAM.xlsx',
    'LB NAGAR': 'LB NAGAR.xlsx',
    'MIYAPUR': 'MIYAPUR.xlsx'
}

# Function to check if username exists
def is_username_taken(username):
    cursor.execute("SELECT COUNT(*) FROM producers_log WHERE username = %s UNION SELECT COUNT(*) FROM consumers_log WHERE username = %s", (username, username))
    counts = cursor.fetchone()
    return counts[0] > 0

# Function to insert user data into the appropriate table
def insert_user(username, name, number, location, password, user_type):
    latitude, longitude = get_lat_long(location)

    if latitude and longitude:
        table_name = "producers_log" if user_type == "Producer" else "consumers_log"
        cursor.execute(f"INSERT INTO {table_name} (username, name, number, location, latitude, longitude, password) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                       (username, name, number, location, latitude, longitude, password))
        mydb.commit()
        st.success(f"User added to {table_name} table.")
    else:
        st.error("Could not fetch latitude and longitude for the provided location.")

# Function to verify login
def verify_login(username, password, user_type):
    table_name = "producers_log" if user_type == "Producer" else "consumers_log"
    cursor.execute(f"SELECT * FROM {table_name} WHERE username = %s AND password = %s", (username, password))
    return cursor.fetchone()  # returns None if no match is found

# Function to fetch user's location, latitude, and longitude based on login
# Function to fetch user info from database
def fetch_user_info(username, user_type):
    if user_type == "Producer":
        cursor.execute("SELECT name, location, latitude, longitude, number FROM producers_log WHERE username = %s", (username,))
        result = cursor.fetchone()
        
        if result:
            name, location, latitude, longitude, phone_number = result
            return name, location, latitude, longitude, phone_number
    return None


# Function to insert produce into the database
def post_produce(name, crop_type, production, price, phone_number, location, latitude, longitude):
    cursor.execute(
        """
        INSERT INTO produce (username, name, crop, quantity, price, phone_number, location, latitude, longitude)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            st.session_state.user_name,
            name,
            crop_type,
            production,
            price,
            phone_number,
            location,
            latitude,
            longitude
        )
    )
    mydb.commit()
    st.success("Produce posted successfully!")


# Function to fetch all producers and consumers for the map
def fetch_all_locations(user_type):
    if user_type == "Producer":
        cursor.execute("SELECT latitude, longitude, username, location FROM producers_log")
    else:
        cursor.execute("SELECT latitude, longitude, username, location FROM consumers_log")
    return cursor.fetchall()

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS
def send_sms(to_number, message):
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=f"+91{to_number}"
    )
            

# Sidebar for navigation
st.sidebar.title("Navigation")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_name = None
    st.session_state.user_type = None

# Main application title
st.title("AgriFlow🪴")

# Page: Login and Signup
if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Select:", ["Login", "Sign Up"])

    # Page: Login
    if menu == "Login":
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        user_type = st.selectbox("User Type", ["Producer", "Consumer"])

        if st.button("Login"):
            user = verify_login(username, password, user_type)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_name = username
                st.session_state.user_type = user_type
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # Page: Sign Up
    elif menu == "Sign Up":
        st.header("Sign Up")

        # Predefined major places in Hyderabad
        hyderabad_places = [
            'Banjara Hills', 'Jubilee Hills', 'Gachibowli', 'Hitech City',
            'Madhapur', 'Kondapur', 'Begumpet', 'Secunderabad', 
            'Charminar', 'Kukatpally', 'Mehdipatnam', 'Attapur',
            'Shamshabad', 'Miyapur', 'Nallagandla', 'Kompally'
        ]

        # Signup form
        with st.form("signup_form"):
            user_type = st.selectbox("User Type", ["Producer", "Consumer"])
            username = st.text_input("Username", max_chars=20)
            name = st.text_input("Name", max_chars=50)
            number = st.text_input("Phone Number", max_chars=10)
            location = st.selectbox("Location (Hyderabad only)", hyderabad_places)
            password = st.text_input("Password", type="password")
            
            submit = st.form_submit_button("Signup")

        # On form submission, validate and insert data
        if submit:
            if username and name and number and location and password:
                if is_username_taken(username):
                    st.error("Username is already taken. Please choose another one.")
                else:
                    insert_user(username, name, number, location, password, user_type)
            else:
                st.error("All fields are required.")

# If logged in, show new sidebar options
if st.session_state.logged_in:
    if st.session_state.user_type=="Producer":
        menu = st.sidebar.selectbox("Select a page:", ["Post Produce", "History","Predictions", "Map","Chatbot","Crop Recommendation", "Logout"])

        # Page: Post Produce
        if menu == "Post Produce":
            user_info = fetch_user_info(st.session_state.user_name, st.session_state.user_type)
            if user_info:
                name, location, latitude, longitude, phone_number = user_info

                st.header("Post Your Produce")
                with st.form("produce_form"):

                    # Input fields for the produce
                    crop_type = st.text_input("Crop Type", max_chars=100)
                    price = st.number_input("Price (per unit)", min_value=0.0, format="%.2f")
                    production = st.number_input("Production Quantity (in units)", min_value=0)

                    submit = st.form_submit_button("Post Produce")

                if submit:
                    if crop_type and price > 0 and production > 0:
                        # Fetch lat/long if they are None
                        if latitude is None or longitude is None:
                            latitude, longitude = get_lat_long(location)
                        post_produce(name, crop_type, production, price, phone_number, location, latitude, longitude)
                        st.success(f"Produce posted successfully: {crop_type}")
                    else:
                        st.error("All fields are required and must have valid values.")
            else:
                st.error("Could not retrieve user info from the database. Please log in again.")


        elif menu == "Crop Recommendation":
            st.subheader("🌾 Crop Recommendation")
            X = data.drop('Crop', axis=1)  # Features: Nitrogen, Phosphorus, etc.
            y = data['Crop']               # Target: Crop to recommend

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale the features for better performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the Random Forest Classifier
            rfc_algo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rfc_algo.fit(X_train_scaled, y_train)

            # Evaluate the model's performance
            y_pred = rfc_algo.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            # Create inputs for user-provided new data
            st.header("Enter Environmental and Soil Data")

            nitrogen = st.number_input("Nitrogen content (N)", min_value=0, max_value=100, value=85)
            phosphorus = st.number_input("Phosphorus content (P)", min_value=0, max_value=100, value=40)
            potassium = st.number_input("Potassium content (K)", min_value=0, max_value=100, value=43)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=22.5)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
            ph_value = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.4)
            rainfall = st.number_input("Rainfall (mm)", min_value=0, max_value=500, value=180)

            # New input data from the user
            new_data = {
                'N': nitrogen,
                'P': phosphorus,
                'K': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph_value,
                'rainfall': rainfall
            }

            # Convert the input into a DataFrame and scale it
            new_data_df = pd.DataFrame([new_data])
            new_data_scaled = scaler.transform(new_data_df)

            # Predict the recommended crop
            if st.button("Predict Recommended Crop"):
                recommended_crop = rfc_algo.predict(new_data_scaled)
                st.success(f"Recommended Crop: {recommended_crop[0]}")

        elif menu == "Predictions":
            st.subheader("🔮 Predict Next Month's Crop Consumption")
            selected_area = st.selectbox('Select an area', list(area_files.keys()))

            # Load the dataset based on the selected area
            file_path = area_files[selected_area]
            df = pd.read_excel(file_path)

            # Clean the data by renaming columns (removing leading/trailing spaces if necessary)
            df.columns = df.columns.str.strip()

            # Group the dataset by 'Crop' to predict for each crop individually
            crops = df['Crop'].unique()

            predictions = []

            # Loop through each crop and predict the next month's consumption based on weekly data
            for crop in crops:
                # Filter the data for the current crop
                crop_data = df[df['Crop'] == crop]

                # Prepare the features (weekly consumption) and target (total consumption for the month)
                X = crop_data[['week 1 (in grams)', 'Week 2 (in grams)', 'Week 3 (in grams)', 'Week 4 (in grams)']]
                y = crop_data['Total (in grams)']

                # Check if there's enough data to train the model
                if len(X) > 1:
                    # Split the dataset into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Create and train the linear regression model
                    lr_algo = LinearRegression()
                    lr_algo.fit(X_train, y_train)

                    # Use the last available month of data to predict the next month's consumption
                    last_month_data = X.tail(1)
                    next_month_prediction = lr_algo.predict(last_month_data)[0]
                    rounded_prediction = round(next_month_prediction)

                    # Append the prediction for the current crop
                    predictions.append({'Crop': crop, 'Predicted Consumption (in grams)': rounded_prediction})

            # Convert predictions to a DataFrame for easy display
            predictions_df = pd.DataFrame(predictions)

            # Display the predicted consumption for each crop for the next month
            st.write(predictions_df)

        # Page: History
        elif menu == "History":
            st.subheader("📜 Your Produce and Consumer History")

            # Selection box for data type
            data_type = st.selectbox("Select data to view:", ["Production Data", "Consumer Data"])

            # Fetch and display production history if user is a Producer
            if data_type == "Production Data" and st.session_state.user_type == "Producer":
                st.write("### Your Production History")
                cursor.execute("SELECT name, crop, quantity, price, phone_number, location FROM produce WHERE username=%s", (st.session_state.user_name,))
                produce_history = cursor.fetchall()

                if produce_history:
                    for produce in produce_history:
                        name, crop, quantity, price, phone_number, location = produce
                        st.write(f"**Crop:** {crop} | **Quantity:** {quantity} units | **Price:** ₹{price}/unit ")
                else:
                    st.write("No production history found.")

            # Fetch and display consumer history if user is a Consumer
            elif data_type == "Consumer Data" and st.session_state.user_type == "Consumer":
                st.write("### Your Consumer History")
                cursor.execute("SELECT * FROM consumers_log WHERE username=%s", (st.session_state.user_name,))
                consumer_history = cursor.fetchall()

                if consumer_history:
                    for consumer in consumer_history:
                        consumer_id, consumer_name, location, latitude, longitude = consumer
                        st.write(f"**Name:** {consumer_name} | **Location:** {location} | **Latitude:** {latitude} | **Longitude:** {longitude}")
                else:
                    st.write("No consumer history found.")

            else:
                st.write("You do not have permission to view this data.")

        # Map
        elif menu == "Map":
            st.subheader("🗺️ Map of Producers and Consumers with Routes")

            # Initialize the map centered at a general location
            map_center = [17.387140, 78.491684]  # Example center point
            m = folium.Map(location=map_center, zoom_start=6)

            # Selection box to choose which locations to display (Consumers, Producers, or Both)
            display_option = st.selectbox("Display Options", ["Producers", "Consumers", "Both"])

            # Fetch producer and consumer locations
            producers, consumers =  fetch_producer_consumer_locations()

            # Initialize variable to store details of clicked marker
            marker_details = None  

            # Display Producers
            if display_option == "Producers" or display_option == "Both":
                for producer in producers:
                    folium.Marker(
                        location=[producer[0], producer[1]],  # Lat/Long from producer data
                        popup=f"<b>Producer</b><br>{producer[2] if len(producer) > 2 else 'Unknown Crop'}<br>{producer[3] if len(producer) > 3 else 'Unknown Location'}",
        # Crop and location
                        icon=folium.Icon(color='red')
                    ).add_to(m)

            # Display Consumers
            if display_option == "Consumers" or display_option == "Both":
                for consumer in consumers:
                    folium.Marker(
                        location=[consumer[0], consumer[1]],  # Lat/Long from consumer data
                        popup=f"<b>Consumer</b><br>{consumer[2] if len(consumer) > 2 else 'Unknown Name'}<br>{consumer[3] if len(consumer) > 3 else 'Unknown Location'}",
        # Name and location
                        icon=folium.Icon(color='blue')
                    ).add_to(m)

            # Capture the click event and display details below the map
            map_data = st_folium(m, width=700, height=500)

            # Check if any marker is clicked, and show its details
            if map_data and 'last_object_clicked' in map_data and map_data['last_object_clicked']:
                clicked_location = map_data['last_object_clicked']['lat'], map_data['last_object_clicked']['lng']

                # Check if the clicked location is from producers or consumers
                for producer in producers:
                    if [producer[0], producer[1]] == list(clicked_location):
                        st.write(f"### Producer Details")
                        st.write(f"**Crop**: {producer[2]}")
                        st.write(f"**Location**: {producer[3]}")
                        break

                for consumer in consumers:
                    if [consumer[0], consumer[1]] == list(clicked_location):
                        st.write(f"### Consumer Details")
                        st.write(f"**Name**: {consumer[2]}")
                        st.write(f"**Location**: {consumer[3]}")
                        break
        # Chatbot
        elif menu == "Chatbot":
            st.subheader("🤖 Chatbot")

            
            def get_pdf_text(pdf_file_path):
                text = ""
                with open(pdf_file_path, "rb") as file:
                    reader = PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text()  # Extract text from each page
                return text

            def get_text_chunks(raw_text):
                text_splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = text_splitter.split_text(raw_text)
                return chunks

            def get_vectorstore(text_chunks):
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                return vectorstore

            def get_conversation_chain(vectorstore):
                llm = ChatOpenAI()
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever(), memory=memory
                )
                return conversation_chain

            def handle_userinput(user_question):
                response = st.session_state.conversation({"question": user_question})

                st.session_state.chat_history = response["chat_history"]
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            def main():
                load_dotenv()
                st.write(css, unsafe_allow_html=True)

                if "conversation" not in st.session_state:
                    st.session_state.conversation = None
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = None

                # Input box for user questions
                user_question = st.text_input("Ask me anything about sustainable food systems!")
                if user_question:
                    handle_userinput(user_question)

                # Load and process the PDF file
                pdf_file_path = "C:\\Users\\USER\\Desktop\\Work\\Projects\\Farming\\health_benefits.pdf"
                raw_text = get_pdf_text(pdf_file_path)

                # Split the text into chunks and create the vectorstore
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain with the vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

            if __name__ == "__main__":
                main()
        # Logout
        elif menu == "Logout":
            st.session_state.logged_in = False
            st.session_state.user_name = None
            st.session_state.user_type = None
            st.success("You have logged out.")
            st.rerun()

#consumer

    else:
        st.sidebar.title("Consumer Options")
        consumer_menu = st.sidebar.selectbox("Select a page:", ["Buy Produce", "History", "Map", "Chatbot" ,"Logout"])

        
        if consumer_menu == "Buy Produce":
            st.header("🛒 Buy Produce")

            # Selection box for areas
            areas = ["Jubilee Hills", "Abids", "Miyapur", "Mehdipatnam", "LB Nagar", "ECIL"]
            selected_area = st.selectbox("Select an area:", areas)

            # Fetch producer data from the database based on the selected area
            cursor.execute("SELECT name, crop, quantity, price, phone_number, location FROM produce WHERE location = %s", (selected_area,))
            producers = cursor.fetchall()

            if producers:
                st.write(f"**Producers in {selected_area}:**")
                for index, producer in enumerate(producers):
                    name, crop, quantity, price, phone, location = producer

                    st.write(f"**Name:** {name}")
                    st.write(f"**Crop:** {crop}")
                    st.write(f"**Quantity Available:** {quantity} kg")
                    st.write(f"**Price per kg:** ₹{price}")
                    st.write(f"**Phone Number:** {phone}")

                    # Order button
                    if st.button(f"Order {name}", key=f"buy_{name}_{index}"):
                        # Store producer info and flag as ordered in session state
                        st.session_state.selected_producer = {
                            "name": name, "crop": crop, "quantity": quantity, "price": price, "location": location
                        }
                        st.session_state.ordering = True

                    # Check if the user has selected this producer to order from
                    if st.session_state.get("ordering") and st.session_state.get("selected_producer", {}).get("name") == name:
                        # Display number input for order quantity
                        order_quantity = st.number_input(
                            f"How many kilograms of {crop} would you like to buy?", 
                            min_value=1, max_value=quantity, key=f"order_{name}_{index}"
                        )
                        st.session_state.order_quantity = order_quantity

                        # Submit order button
                        if st.button(f"Submit Order for {name}", key=f"submit_{name}_{index}"):
                            # Check if order quantity is valid
                            if order_quantity > quantity:
                                st.error(f"Error: Requested quantity exceeds available stock! Only {quantity} kg available.")
                            elif order_quantity <= 0:
                                st.error("Error: Please enter a valid quantity.")
                            else:
                                total_price = order_quantity * price  # Calculate total price
                                # Insert order into the orders table
                                cursor.execute(
                                    "INSERT INTO orders (consumer_username, producer_username, crop, quantity, total_price) VALUES (%s, %s, %s, %s, %s)",
                                    (st.session_state.user_name, name, crop, order_quantity, total_price)
                                )

                                # Update quantity in the produce table
                                new_quantity = quantity - order_quantity
                                cursor.execute("UPDATE produce SET quantity = %s WHERE name = %s AND crop = %s AND location = %s", 
                                            (new_quantity, name, crop, location))

                                # Fetch buyer's phone number from the consumers_log table
                                cursor.execute("SELECT number FROM consumers_log WHERE username = %s", (st.session_state.user_name,))
                                buyer_phone = cursor.fetchone()[0]

                                # Commit the transaction
                                mydb.commit()

                                # Send SMS to buyer
                                buyer_message = f"Dear {st.session_state.user_name}, your order for {order_quantity} kg of {crop} has been confirmed. Total price: ₹{total_price}."
                                send_sms(buyer_phone, buyer_message)

                                # Send SMS to seller
                                seller_message = f"You have a new order! {order_quantity} kg of {crop} ordered by {st.session_state.user_name} in {selected_area}. Please contact them for further details."
                                send_sms(phone, seller_message)

                                st.success(f"Order placed successfully for {order_quantity} kg of {crop}. Updated stock: {new_quantity} kg.")
                                st.success("Messages have been sent to both the buyer and the seller.")

                    st.write("---")  # Separator for producers
            else:
                st.write(f"No producers available in {selected_area}.")


        elif consumer_menu == "History":
            # Implement functionality to view request history here

            st.header("📜 Order History")

            # Fetch order history for the logged-in user
            cursor.execute("SELECT producer_username, crop, quantity, total_price, order_date FROM orders WHERE consumer_username = %s ORDER BY order_date DESC", 
                        (st.session_state.user_name,))
            orders = cursor.fetchall()

            if orders:
                st.write(f"**Order history for {st.session_state.user_name}:**")

                for order in orders:
                    producer_username, crop, quantity, total_price, order_date = order

                    st.write(f"**Producer:** {producer_username}")
                    st.write(f"**Crop:** {crop}")
                    st.write(f"**Quantity:** {quantity} kg")
                    st.write(f"**Total Price:** ₹{total_price}")
                    st.write(f"**Order Date:** {order_date}")

                    st.write("---")  # Separator for each order
            else:
                st.write("No order history available.")

        elif consumer_menu == "Map":
            st.subheader("🗺️ Map of Producers and Consumers with Routes")

            # Initialize the map centered at a general location
            map_center = [17.387140, 78.491684]  # Example center point
            m = folium.Map(location=map_center, zoom_start=6)

            # Selection box to choose which locations to display (Consumers, Producers, or Both)
            display_option = st.selectbox("Display Options", ["Producers", "Consumers", "Both"])

            # Fetch producer and consumer locations
            producers, consumers = fetch_producer_consumer_locations()

            # Initialize variable to store details of clicked marker
            marker_details = None  

            # Display Producers
            if display_option == "Producers" or display_option == "Both":
                for producer in producers:
                    folium.Marker(
                        location=[producer[0], producer[1]],  # Lat/Long from producer data
                        popup=f"<b>Producer</b><br>{producer[2] if len(producer) > 2 else 'Unknown Crop'}<br>{producer[3] if len(producer) > 3 else 'Unknown Location'}",
        # Crop and location
                        icon=folium.Icon(color='red')
                    ).add_to(m)

            # Display Consumers
            if display_option == "Consumers" or display_option == "Both":
                for consumer in consumers:
                    folium.Marker(
                        location=[consumer[0], consumer[1]],  # Lat/Long from consumer data
                        popup=f"<b>Consumer</b><br>{consumer[2] if len(consumer) > 2 else 'Unknown Name'}<br>{consumer[3] if len(consumer) > 3 else 'Unknown Location'}",
        # Name and location
                        icon=folium.Icon(color='blue')
                    ).add_to(m)

            # Capture the click event and display details below the map
            map_data = st_folium(m, width=700, height=500)

            # Check if any marker is clicked, and show its details
            if map_data and 'last_object_clicked' in map_data and map_data['last_object_clicked']:
                clicked_location = map_data['last_object_clicked']['lat'], map_data['last_object_clicked']['lng']

                # Check if the clicked location is from producers or consumers
                for producer in producers:
                    if [producer[0], producer[1]] == list(clicked_location):
                        st.write(f"### Producer Details")
                        st.write(f"**Crop**: {producer[2]}")
                        st.write(f"**Location**: {producer[3]}")
                        break

                for consumer in consumers:
                    if [consumer[0], consumer[1]] == list(clicked_location):
                        st.write(f"### Consumer Details")
                        st.write(f"**Name**: {consumer[2]}")
                        st.write(f"**Location**: {consumer[3]}")
                        break

        elif consumer_menu == "Chatbot":
            st.subheader("🤖 Chatbot")

            
            def get_pdf_text(pdf_file_path):
                text = ""
                with open(pdf_file_path, "rb") as file:
                    reader = PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text()  # Extract text from each page
                return text

            def get_text_chunks(raw_text):
                text_splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
                )
                chunks = text_splitter.split_text(raw_text)
                return chunks

            def get_vectorstore(text_chunks):
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                return vectorstore

            def get_conversation_chain(vectorstore):
                llm = ChatOpenAI()
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever(), memory=memory
                )
                return conversation_chain

            def handle_userinput(user_question):
                response = st.session_state.conversation({"question": user_question})

                st.session_state.chat_history = response["chat_history"]
                for i, message in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            def main():
                load_dotenv()
                st.write(css, unsafe_allow_html=True)

                if "conversation" not in st.session_state:
                    st.session_state.conversation = None
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = None

                # Input box for user questions
                user_question = st.text_input("Ask me anything about sustainable food systems!")
                if user_question:
                    handle_userinput(user_question)

                # Load and process the PDF file
                pdf_file_path = "C:\\Users\\USER\\Desktop\\Work\\Projects\\Farming\\health_benefits.pdf"
                raw_text = get_pdf_text(pdf_file_path)

                # Split the text into chunks and create the vectorstore
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain with the vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

            if __name__ == "__main__":
                main()

        elif consumer_menu == "Logout":
            st.session_state.logged_in = False
            st.session_state.user_name = None
            st.session_state.user_type = None
            st.success("You have logged out.")
            st.rerun()