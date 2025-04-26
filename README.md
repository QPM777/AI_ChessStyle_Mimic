# ChessStyle Mimic

This project aims to mimic chess moves based on the playing style of dif ferent players.

## How to run the project

1. **Generate the dataset**

   We have already a csv to try but in other case download a .pgn and run the provided notebook (`generate_dataset.ipynb`) to create the required CSV file.

2. **Train and run the model**

   After generating the CSV, run the following command to execute the model:

   ```bash
   modal run model.py::main
   ```
