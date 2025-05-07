import os


def main():
    print("Choose tracker:")
    print("1.Frame differencing")
    print("2.Kalman simple tracker")
    print("3.Observation-Centric SORT")
    choice = input("Enter 1/2/3: ")

    if choice == "1":
        os.system("python frame_differencing_tracker/main.py")
    elif choice == "2":
        os.system("python kalman_simple_tracker/main.py")
    elif choice == "3":
        os.system("python oc_sort/main.py")
    else:
        print("Wrong input.")

if __name__ == "__main__":
    main()