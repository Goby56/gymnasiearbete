with open("analysis/models.txt") as file:
    for name in file:
        __import__(name).main()