import os

class datawriter():


    def __init__(self, store_path = "results/"):
        id = 0
        while(os.path.exists(store_path + str(id) + "_info.txt")):
            id += 1

        self.loss_file = open(store_path + str(id) + "_loss.csv", "w")
        self.info_file = open(store_path + str(id) + "_info.txt", "w")
        self.para_file = open(store_path + str(id) + "_para.csv", "w")

        self.info_file.write("Try: {} \n".format(id))
        self.loss_file.write("Epoch, Avg Loss\n")
        self.para_file.write("Epoch, Params\n")

    
    def write_hyperparameters(self, hyperparameters):
        self.info_file.write("--- Hyperparameters: ---\n")
        for key, value in hyperparameters.items():
            self.info_file.write(f"{key:<15}: {str(value):>15}\n")
        self.info_file.write("--- END Hyperparameters ---\n")


    def set_model(self, model):
        self.model = model
        self.info_file.write("Model used: {}\n".format(model))

        self.info_file.write("--- Parameters: ---\n")
        for name, param in model.named_parameters():
            self.info_file.write(f"{name:<15}: {str(param):>15}\n")
        self.info_file.write("--- END Parameters ---\n")


    def write_eval(self, epoch, eval):
        self.loss_file.write(str(epoch) + ", " + str(eval["eval_loss"]) + "\n")

        params = ""
        for name, param in self.model.named_parameters():
            params += str(param) + ", "
        self.para_file.write(str(epoch) + ", " + params[0:-2] + "\n")


    def write_info(self, string):
        self.info_file.write(string + "\n")


    def close(self):
        self.info_file.write("Done! Files are being closed.")
        self.info_file.close()
        self.loss_file.close()
        self.para_file.close()
