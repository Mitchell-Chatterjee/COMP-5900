import os, glob

if __name__ == "__main__":
    my_dir = 'data/tcav/image/concepts'
    concept_folders = [x[0] for x in os.walk(my_dir)][1:]
    print(concept_folders)

    for concept in concept_folders:
        for fname in os.listdir(concept):
            if fname.endswith("color.png"):
                os.remove(os.path.join(concept, fname))