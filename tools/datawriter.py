import os

import data_tools

class datawriter():


    def __init__(self, device="device not known", hyperparameters={}, store_path = "results/"):
        id = 0
        while(os.path.exists(store_path + str(id) + "_info.txt")):
            id += 1

        self.write_params = hyperparameters["write_params"]
        self.write_images = hyperparameters["write_images"]

        self.loss_file = open(store_path + str(id) + "_loss.csv", "w")
        self.info_file = open(store_path + str(id) + "_info.txt", "w")
        if(self.write_params): self.para_file = open(store_path + str(id) + "_para.csv", "w")
        if(self.write_images):
            self.image_path = store_path + str(id) + "_images/"
            os.mkdir(self.image_path)

        self.info_file.write("Try: {} \n".format(id))
        self.loss_file.write("Epoch, Avg Loss\n")
        if(self.write_params): self.para_file.write("Epoch, Params\n")
        self.info_file.write("Using {} device. \n".format(device))
        self.write_hyperparameters(hyperparameters)

    
    def write_hyperparameters(self, hyperparameters):
        self.info_file.write("\n--- Hyperparameters: ---\n")
        for key, value in hyperparameters.items():
            self.info_file.write(f"{key:<20}: {str(value):>15}\n")
        self.info_file.write("--- END Hyperparameters ---\n")


    def set_model(self, model, load_model):
        self.model = model
        self.info_file.write("\nModel used: {}\n".format(model))
        self.info_file.write("Model loaded from 'model.pth' = {} \n".format(load_model))

        if(self.write_params):
            self.info_file.write("\n--- Parameters: ---\n")
            for name, param in model.named_parameters():
                self.info_file.write(f"{name:<15}: {str(param):>15}\n")
            self.info_file.write("--- END Parameters ---\n")


    def write_eval(self, epoch, eval):
        self.loss_file.write(str(epoch) + ", " + str(eval["eval_loss"]) + "\n")

        if(self.write_params):
            params = ""
            for name, param in self.model.named_parameters():
                params += str(param) + ", "
            
            params_string = params[0:-2].replace("\n", "").replace("\r", "").replace(
                "\t", "").replace(",", " / ").replace("  ","")
            self.para_file.write(str(epoch) + ", " + "(" + params_string + ")" + "\n")
        
        if(self.write_images):
            curr_image_path = self.image_path + str(epoch) + "_epoch/"
            os.mkdir(curr_image_path)
            
            images = eval["sample"]["image"]
            predictions = eval["prediction"]
            ground_truth = eval["sample"]["ground_truth"]
            
            batch_size = len(images)

            for i in range(batch_size):
                f = open(curr_image_path + str(eval["sampel"]["original_id"][i] + ".txt"))
                f.close()
                data_tools.store_tensor_as_image(images[i], curr_image_path + str(i) + "_image")
                data_tools.store_tensor_as_image(predictions[i], curr_image_path + str(i) + "_prediction")
                data_tools.store_tensor_as_image(ground_truth[i], curr_image_path + str(i) + "_ground_truth")


    def write_info(self, string):
        self.info_file.write(string + "\n")


    def close(self):
        self.info_file.write("\nDone! Files are being closed.")
        self.info_file.close()
        self.loss_file.close()
        self.para_file.close()
