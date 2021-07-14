import os

class datawriter():


    def __init__(self, store_path = "results/train"):
        id = 0
        while(os.path.exists(store_path + "/info_" + id + ".txt")):
            id += 1

        self.info_file = open(store_path + "/info_" + id + ".txt", "w")
        self.loss_file = open(store_path + "/loss_" + id + ".csv", "w")
        self.para_file = open(store_path + "/para_" + id + ".csv", "w")

        self.info_file.write("Try: {}".format(id))
        self.loss_file.write("Epoch, Avg Loss")
        self.para_file.write("Epoch, Params")

    
    def write_hyperparameters(self, hyperparameters):
        self.info_file.write("--- Hyperparameters: ---")
        for key, value in hyperparameters.items():
            self.info_file.write(f"{key:<15}: {str(value):>15}")
        self.info_file.write("--- END Hyperparameters ---")


    def set_model(self, model):
        self.model = model
        self.info_file.write("Model used: {}".format(model))

        self.info_file.write("--- Parameters: ---")
        for name, param in model.named_parameters():
            self.info_file.write(f"{name:<15}: {str(param):>15}")
        self.info_file.write("--- END Parameters ---")


    def write_eval(self, epoch, eval):
        self.loss_file.write(str(epoch) + ", " + str(eval["eval loss"]))

        params = ""
        for name, param in self.model.named_parameters():
            params += str(param) + ","
        self.para_file.write(str(epoch) + ", " + params)


    def write_info(self, string):
        self.info_file.write(string)


    def close(self):
        self.info_file.write("Done! Files are being closed.")
        self.info_file.close()
        self.loss_file.close()
        self.para_file.close()
