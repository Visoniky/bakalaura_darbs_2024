import sky_methods_library

go_on = True

while go_on:
    while True:
        i = 1

        print("Methods list:")
        for name in sky_methods_library.ALL_NAMES:
            print(f"{i}. {name}")
            i += 1
        method_num = int(input("Choose method: "))

        if method_num < 1 or method_num > len(sky_methods_library.ALL_NAMES):
            print(f"\nCarefully! You can enter only figure in the range from 1 to {len(sky_methods_library.ALL_NAMES)}\n")
        else:
            break

    filter = sky_methods_library.Method(method_num)

    path = input("Folder path: ")
    filter.set_folder_path(path)

    filter.process_all_images()

    filter.print_info()

    while True:
        answer_continue = input("Do you want to process folder one more time? y/n (yes/no)")

        if answer_continue not in ["y", "n"]:
            print("Check your input and try one more time!")
        else:
            if answer_continue == "n":
                go_on = False

            break