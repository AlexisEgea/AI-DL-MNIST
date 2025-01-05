#!/bin/bash

echo  "-----------------------------------------------------------------------------"
echo "|              © Execution Project Draw Predict Digit with CNN               |"
echo "| Author : Alexis EGEA                                                       |"
echo "-----------------------------------------------------------------------------"
echo

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	PYTHON_CMD=python3
elif [[ "$OSTYPE" == "cygwin"* || "$OSTYPE" == "msys"* ]]; then
 	PYTHON_CMD=python
else
	echo "Unsupported OS '$OSTYPE'"
	exit 1
fi

echo "Please select the script you want to run:"
echo "1) main: Use saved model to predict a drawn digit"
echo "2) main_train_model_with_dataset: Create, train, and save a CNN with your dataset"
echo "3) main_train_model_with_mnist: Create, train, and save a CNN with the MNIST dataset"
echo "4) create_mnist_dataset: Create your personal dataset"
read -p "Enter the number of your choice: " CHOICE

case $CHOICE in
	1)
	  echo "main selected"
		SCRIPT="main"
		;;
	2)
	  echo "main_train_model_with_dataset selected"
		SCRIPT="main_train_model_with_dataset"
		;;
	3)
	  echo "main_train_model_with_mnist selected"
		SCRIPT="main_train_model_with_mnist"
		;;
	4)
	  echo "create_mnist_dataset selected"
		SCRIPT="create_mnist_dataset"
		;;
	*)
		echo "Invalid choice. Please rerun the script and enter a valid number."
		exit 1
		;;
esac
echo "-----------------------------------------------------------------------------"
echo

cd src/
$PYTHON_CMD -m $SCRIPT
cd ..

echo
read -p "Press any key to close the terminal window"
