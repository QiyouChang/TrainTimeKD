import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

class myKDBaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        distil_weight=0.5,
        device="cpu",
        log=False,
        logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if device == "cpu":
            self.device = torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print(
                    "Either an invalid device or CUDA is not available. Defaulting to CPU."
                )
                self.device = torch.device("cpu")

        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.ce_fn = nn.CrossEntropyLoss().to(self.device)

    def train_model(
        self,
        epochs_teacher=20,
        epochs_student=10,
        plot_losses=True,
        save_teacher_model=True,
        save_student_model=True,
        save_teacher_model_pth="./models/teacher.pt",
        save_student_model_pth="./models/student.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.teacher_model.train()
        loss_arr_teacher = []
        loss_arr_student = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc_teacher = 0.0
        best_acc_student = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())
        
        save_teacher_dir = os.path.dirname(save_teacher_model_pth)
        save_student_dir = os.path.dirname(save_student_model_pth)
        if not os.path.exists(save_teacher_dir):
            os.makedirs(save_teacher_dir)
        if not os.path.exists(save_student_dir):
            os.makedirs(save_student_dir)

        epoch_ratio = epochs_teacher // epochs_student

        for ep in range(epochs_teacher):
            print("Training Teacher... ")
            epoch_loss = 0.0
            correct = 0
            i = 0

            for (data, label) in self.train_loader:
                i += 1
                print("The", str(i), "th iteration with label:", label)
                if i >= 3:
                    break 
                data = data.to(self.device)
                label = label.to(self.device)
                out = self.teacher_model(data)

                if isinstance(out, tuple):
                    out = out[0]

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.ce_fn(out, label)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss += loss.item()

            epoch_acc = correct / length_of_dataset

            epoch_val_acc = self.evaluate(teacher=True)

            if epoch_val_acc > best_acc_teacher:
                best_acc_teacher = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                wandb.log({"teacher_train_loss": epoch_loss, "teacher_train_acc": epoch_acc, "teacher_test_acc": epoch_val_acc, "teacher_epoch": ep})

            loss_arr_teacher.append(epoch_loss)
            print(
                "Teacher Epoch: {}, Loss: {}, Accuracy: {}".format(
                    ep + 1, epoch_loss, epoch_acc
                )
            )
            if ep // epoch_ratio * epoch_ratio == ep:
                print("Training students...")
                self.teacher_model.eval()
                self.student_model.train()
                student_epoch_loss = 0.0 
                student_correct = 0 
                i = 0 
                for (data, label) in self.train_loader:
                    i += 1
                    print("The", str(i), "th iteration with label:", label)
                    if i >= 3:
                        break 
                    data = data.to(self.device)
                    label = label.to(self.device)

                    student_out = self.student_model(data)
                    teacher_out = self.teacher_model(data)

                    loss = self.calculate_kd_loss(student_out, teacher_out, label)

                    if isinstance(student_out, tuple):
                        student_out = student_out[0]

                    pred = student_out.argmax(dim=1, keepdim=True)
                    student_correct += pred.eq(label.view_as(pred)).sum().item()

                    self.optimizer_student.zero_grad()
                    loss.backward()
                    self.optimizer_student.step()

                    student_epoch_loss += loss.item()

                epoch_acc = student_correct / length_of_dataset

                _, epoch_val_acc = self._evaluate_model(self.student_model, verbose=True)

                if epoch_val_acc > best_acc_student:
                    best_acc_student = epoch_val_acc
                    self.best_student_model_weights = deepcopy(
                        self.student_model.state_dict()
                    )

                if self.log:
                    wandb.log({"student_train_loss": epoch_loss, "student_train_acc": epoch_acc, "student_test_acc": epoch_val_acc, "student_epoch": ep})

                loss_arr_student.append(epoch_loss)
                print(
                    "Student Epoch: {}, Loss: {}, Accuracy: {}".format(
                        ep + 1, epoch_loss, epoch_acc
                    )
                )
            
        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_teacher_model:
            torch.save(self.teacher_model.state_dict(), save_teacher_model_pth)
        if save_student_model:
            torch.save(self.student_model.state_dict(), save_student_model_pth)
        # if plot_losses:
        #     plt.plot(loss_arr_teacher)
        #     plt.plot(loss_arr_student)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=True):
        """
        Evaluate the given model's accuaracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        correct = 0
        outputs = []
        print("in eval model:")
        with torch.no_grad():
            i = 0
            for data, target in self.val_loader:
                i += 1
                print("The", str(i), "th iteration with target:", target)
                if i >= 3:
                    break
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)

                if isinstance(output, tuple):
                    output = output[0]
                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {}".format(accuracy))
        return outputs, accuracy

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        _, accuracy = self._evaluate_model(model)

        return accuracy

    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))
