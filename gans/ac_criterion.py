from copy import deepcopy
from random import randint
import torch
import torch.nn.functional as F
import numpy as np

import ipdb


class ACGANCriterion:
    r"""
    Class implementing all tools necessary for a GAN to take into account class
    conditionning while generating a model (cf Odena's AC-GAN)
    https://arxiv.org/pdf/1610.09585.pdf
    """

    def __init__(self,
                 attribKeysOrder,
                 soft_labels=False,
                 skipAttDfake=[]):
        r"""
        Args:

            attribKeysOrder (dict): dictionary containing the all the possible
                                    categories of the DCGAN model and their
                                    order.

                                    Each entry of the attribKeysOrder is
                                    another dictionary with two fields:

                                    order: order of the category in the input
                                           vector
                                    values: possible values taken by this
                                            category

                                    Such a dictionary is returned by
                                    models.datasets.attrib_dataset.AttribDataset.getKeyOrders()
            skipAttDfake (list):    list containing the indices of attributes to be skipped when
                                    calculating the loss for generated data.
            Ex:
                attribKeysOrder = {"Gender": {"order": 0, "values":["M", "W"]},
                                  "Nationality": {"order": 1,
                                                  "values":["english",
                                                            "french",
                                                            "indian"]}
                                   }
                allowMultiple = ["Nationality"]

                Then a category vector corresponding to this pair could be:
                V = [0, 1, 1, 1, 0]

                Which would correspond to a sample of gender "W" and
                nationalities "english" and "french"
        """
        self.nAttrib = len(attribKeysOrder)
        self.attribSize = [0 for i in range(self.nAttrib)]
        self.keyOrder = ['' for x in range(self.nAttrib)]
        self.labelsOrder = {}

        self.allowMultiple = ["properties"]

        self.inputDict = deepcopy(attribKeysOrder)
        self.skipAttDfake = skipAttDfake

        self.keyOrder = list(attribKeysOrder.keys())
        
        self.attribSize = []
        self.att_loss = []
        for i, att in enumerate(attribKeysOrder):
            # check which loss to apply dependeing on typ

            if attribKeysOrder[att]['loss'] == 'mse':
                # if float we use MSE: regression
                self.attribSize.append(len(attribKeysOrder[att]["values"]))
                self.att_loss.append('mse')
            elif attribKeysOrder[att]['loss'] == 'bce':
                self.attribSize.append(len(attribKeysOrder[att]["values"]))
                self.labelsOrder[att] = {index: label for label, index in
                                     enumerate(attribKeysOrder[att]["values"])}
                self.att_loss.append('bce')    
            elif attribKeysOrder[att]['loss'] == 'xentropy':
                self.attribSize.append(len(attribKeysOrder[att]["values"]))
                self.labelsOrder[att] = {index: label for label, index in
                                     enumerate(attribKeysOrder[att]["values"])}
                self.att_loss.append('xentropy')
            else:
                raise KeyError(f"Loss mode {attribKeysOrder[att]['loss']} not understood")

        self.labelWeights = torch.tensor(
            [1.0 for x in range(self.getInputDim())])

        for i, key in enumerate(attribKeysOrder):
            if attribKeysOrder[key].get('weights', None) is not None:
                shift = sum(self.attribSize[:i])
                for value, weight in attribKeysOrder[key]['weights'].items():
                    self.labelWeights[shift +
                                      self.labelsOrder[key][value]] = weight

        self.sizeOutput = self.nAttrib
        self.soft_labels = soft_labels

    def generateConstraintsFromVector(self, n, labels):

        vect = []

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            key = self.keyOrder[i]
            if key in labels:
                value = labels[key]
                index = self.labelsOrder[key][value]
                out = torch.zeros(n, C, 1, 1)
                out[:, index] = 1
            else:
                v = np.random.randint(0, C, n)
                w = np.zeros((n, C), dtype='float32')
                w[np.arange(n), v] = 1
                out = torch.tensor(w).view(n, C, 1, 1)

            vect.append(out)
        return torch.cat(vect, dim=1)

    def buildRandomCriterionTensor(self, sizeBatch, skipAtts=False):
        r"""
        Build a batch of vectors with a random combination of the values of the
        existing classes

        Args:
            sizeBatch (int): number of vectors to generate

        Return:
            targetVector, latentVector

            targetVector : [sizeBatch, M] tensor used as a reference for the
                           loss computation (see self.getLoss)
            latentVector : [sizeBatch, M', 1, 1] tensor. Should be
                           concatenatenated with the random GAN input latent
                           veCtor

            M' > M, input latent data should be coded with one-hot inputs while
            pytorch requires a different format for softmax loss
            (see self.getLoss)
        """
        targetOut = []
        inputLatent = []

        for i in range(self.nAttrib):
            if self.keyOrder[i] in self.skipAttDfake and skipAtts: continue
            
            C = self.attribSize[i]
            v = np.random.randint(0, C, sizeBatch)
            w = np.zeros((sizeBatch, C), dtype='float32')
            w[np.arange(sizeBatch), v] = 1
            y = torch.tensor(w).view(sizeBatch, C)

            inputLatent.append(y)
            targetOut.append(torch.tensor(v).float().view(sizeBatch, 1))
        return torch.cat(targetOut, dim=1), torch.cat(inputLatent, dim=1)

    def buildLatentCriterion(self, targetCat, skipAtts=False):

        if skipAtts:
            total_att_size = 0
            for i, key in enumerate(self.keyOrder):
                if key not in self.skipAttDfake:
                    total_att_size += self.attribSize[i]
        else:
            total_att_size = sum(self.attribSize)
        
        batchSize = len(targetCat)
        idx = torch.arange(batchSize, device=targetCat.device)
        targetOut = torch.zeros((batchSize, total_att_size), device=targetCat.device)
        shift = 0
        shift2 = 0
        for i in range(self.nAttrib):
            if skipAtts and self.keyOrder[i] in self.skipAttDfake: 
                if self.att_loss[i] in ['xentropy', 'sxentropy']:
                    shift2 += 1
                elif self.att_loss[i] in ['mse', 'bce']:
                    shift2 += self.attribSize[i]
                shift += self.attribSize[i]
                continue

            if self.att_loss[i] in ['xentropy', 'sxentropy']:
                targetOut[idx, (shift + targetCat[:, shift2]).int().tolist()] = 1
                shift2 += 1
            elif self.att_loss[i] in ['mse', 'bce']:
                targetOut[idx, shift:shift + self.attribSize[i]] = \
                    targetCat[:, shift2:shift2 + self.attribSize[i]]
                shift2 += self.attribSize[i]
            # elif self.att_loss[i] == 'bce':
            #     ipdb.set_trace()
            shift += self.attribSize[i]

        return targetOut

    def getInputDim(self, G_latent_dim=False):
        r"""
        Size of the latent vector given by self.buildRandomCriterionTensor
        """
        if G_latent_dim:
            total_att_size = 0
            for i, key in enumerate(self.keyOrder):
                if key not in self.skipAttDfake:
                    total_att_size += self.attribSize[i]
            return total_att_size
        else:
            return sum(self.attribSize)

    def getPredictionLabels(self, outputD):

        shiftInput = 0
        # remove Wgan distance
        outputD = outputD[:, :-1]
        outIdx = torch.Tensor(size=(outputD.size(0), 0))
        outIdx = outIdx.to(outputD.device)
        outActivation = []

        for i in range(self.nAttrib):
            C = self.attribSize[i]
            locInput = outputD[:, shiftInput:(shiftInput+C)]
            if self.att_loss[i] == 'xentropy':

                locPred = torch.softmax(locInput, dim=1)
                tmp = torch.argmax(locPred, dim=1, keepdim=True).float()
                outIdx = torch.cat([outIdx, tmp], dim=1)
            elif self.att_loss[i] in ['bce', 'mse']:
                locPred = locInput
                locPred = torch.sigmoid(locInput)
                outIdx = torch.cat([outIdx, locPred], dim=1)
            outActivation.append(locPred)
            shiftInput += C
        return outIdx, outActivation

    def soft_cross_entropy(self, pred, target, lprob=0.3, hprob=(0.7, 1.2)):
        n_cls = pred.size(1)
        target_ = torch.rand(pred.size()) * lprob
        for i, j in enumerate(target):
            target_[i, j] = torch.rand(1)*(hprob[1] - hprob[0]) + hprob[0]
        return -(target_ * torch.nn.functional.log_softmax(pred, -1)).sum(dim=1).mean()

    def getCriterion(self, outputD, target, skipAtts=False):
        r"""
        Compute the conditional loss between the network's output and the
        target. This loss, L, is the sum of the losses Lc of the categories
        defined in the criterion. We have:

                 | Cross entropy loss for the class c if c is attached to a
                   classification task.
            Lc = | Multi label soft margin loss for the class c if c is
                   attached to a tagging task
        """
        loss = 0
        shiftInput = 0
        shiftTarget = 0
        self.labelWeights = self.labelWeights.to(outputD.device)
        losses = []
        for i in range(self.nAttrib):
            C = self.attribSize[i]
            if self.keyOrder[i] not in self.skipAttDfake or not skipAtts:

                locInput = outputD[:, shiftInput:(shiftInput+C)]
                
                # if self.keyOrder[i] in self.allowMultiple:
                #     locTarget = target[:, shiftTarget:(shiftTarget+C)]
                #     locLoss = F.multilabel_soft_margin_loss(locInput, locTarget.long())
                #     shiftTarget += C
                # else:
                #     if self.soft_labels:
                #         locLoss = self.soft_cross_entropy(locInput, locTarget)

                if self.att_loss[i] == 'mse':
                    locTarget = target[:, shiftTarget:shiftTarget + C]
                    locInput = torch.sigmoid(locInput)
                    locTarget = locTarget.reshape(locInput.size())
                    locLoss = F.mse_loss(locInput, locTarget)
                    shiftTarget += C
                elif self.att_loss[i] == 'bce':
                    locTarget = target[:, shiftTarget:shiftTarget + C]
                    locInput = torch.sigmoid(locInput)
                    locLoss = F.binary_cross_entropy(locInput, locTarget)
                    shiftTarget += C
                else:
                    locTarget = target[:, shiftTarget]
                    locLoss = F.cross_entropy(locInput, locTarget.long(), 
                                          weight=self.labelWeights[shiftInput:(shiftInput+C)])
                    shiftTarget += 1
                loss += locLoss
                losses.append((locLoss, self.keyOrder[i]))
            
            shiftInput += C
        # return losses
        return loss
