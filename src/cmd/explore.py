from src.utils import explore_dataset

def main():
    info = explore_dataset()
    print("Total samples:", info["total_samples"])
    print("Class counts (0 to 6):", info["class_counts"])

if __name__ == "__main__":
    main()
