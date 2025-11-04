# Summary

This project is to provide a proof of concept for creating a abnormality detection on Radiometric data. This is used when using a thermographic camera to detect abnormal temperature patterns in a scene.

## Features

- A simulation of a thermographic camera capturing radiometric data using a simulation that the system is reading the thermographic data from the sun. It should be assumed that the source of this data will change to the raw input from a thermographic camera in the future. The system should stream into the engine radiometric data in real time with minor fluctuations to simulate real world conditions (such as clouds passing in front of the sun, etc). These should play like small waves of radiometric energy changes over time that are visible in the heatmap visualization but within realistic limits.
- A visualization of the thermographic data in a 2D heatmap format that is displayed in an application.
- A tool for capturing the baseline thermographic data for a scene that consists of multiple parameters, for example: time of day, weather conditions, and other environmental factors. For the proof of concept, in the situation that we are capturing the sun, we will use the time of day as the only parameter assuming that the sun is giving off less radiometric energy in the morning and evening, and more during midday.
- An abnormality detection algorithm that compares the current thermographic data against the baseline data to identify significant deviations that may indicate an abnormality in the scene.
- A way to capture the Areas of interest where radiometric abnormalities are detected for further analysis. In this proof of concept we will simulate Sunspots or Solar Flares as the abnormalities and make them occur in real time in the simulation. The simulation will have tools to force these 'events' to happen by streaming in changes in the radiometric data in real time for the engine to detect and capture.
- When abnormalities are detected, the system will log the event with relevant details such as timestamp, location in the scene, and severity of the abnormality. In the user interface, it will also HIGHLIGHT the areas where abnormalities are detected on the heatmap visualization.

## Technical Stack

- Programming Language: Python
- Libraries: NumPy, Matplotlib, OpenCV
- Frameworks: Flask (for web application)
- Database: SQLite (for storing baseline data and detected abnormalities)
- Simulation Tools: Custom simulation scripts for generating radiometric data

## Technical Considerations

- Hardware Requirements: Thermographic Camera input (Raspberry Pi compatible camera for future implementation)

## Project Phases

1. Research and Planning
   - Understand the principles of thermography and radiometric data.
   - Define the requirements and scope of the project.
2. Data Simulation
   - Develop a simulation to generate radiometric data.
3. Visualization
   - Create a 2D heatmap visualization of the radiometric data that updates over time.
4. Baseline Data Capture
   - Implement a tool to capture and store baseline thermographic data.
5. Abnormality Detection Algorithm
   - Develop and test the abnormality detection algorithm.
6. User Interface
   - Build a user interface to display the detected abnormalities.
