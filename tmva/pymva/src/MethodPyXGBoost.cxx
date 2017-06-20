// @(#)root/tmva/pymva $Id$
// Author: Enhao Song, 2017

#include <Python.h>
#include "TMVA/MethodPyXGBoost.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "TMVA/Types.h"
#include "TMVA/Config.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Results.h"
#include "TMVA/TransformationHandler.h"
#include "TMVA/VariableTransformBase.h"

using namespace TMVA;

REGISTER_METHOD(PyXGBoost)

ClassImp(MethodPyXGBoost)

MethodPyXGBoost::MethodPyXGBoost(const TString &jobName, const TString &methodTitle, DataSetInfo &dsi, const TString &theOption)
   : PyMethodBase(jobName, Types::kPyXGBoost, methodTitle, dsi, theOption) {
   fNumBoostRound = 10;
   fVerbose = 1;
   fContinueTraining = false;
   fSaveBestOnly = true;
   fTriesEarlyStopping = -1;
   fLearningRateSchedule = ""; // empty string deactivates learning rate scheduler
   fFilenameTrainedModel = ""; // empty string sets output model filename to default (in weights/)
}

MethodPyXGBoost::MethodPyXGBoost(DataSetInfo &theData, const TString &theWeightFile)
    : PyMethodBase(Types::kPyXGBoost, theData, theWeightFile) {
   fNumBoostRound = 10;
   fVerbose = 1;
   fContinueTraining = false;
   fSaveBestOnly = true;
   fTriesEarlyStopping = -1;
   fLearningRateSchedule = ""; // empty string deactivates learning rate scheduler
   fFilenameTrainedModel = ""; // empty string sets output model filename to default (in weights/)
}

MethodPyXGBoost::~MethodPyXGBoost() {
}

Bool_t MethodPyXGBoost::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t) {
   if (type == Types::kRegression) return kTRUE;
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass && numberClasses >= 2) return kTRUE;
   return kFALSE;
}

///////////////////////////////////////////////////////////////////////////////

void MethodPyXGBoost::DeclareOptions() {
   DeclareOptionRef(fFilenameModel, "FilenameModel", "Filename of the initial XGBoost model");
   DeclareOptionRef(fFilenameTrainedModel, "FilenameTrainedModel", "Filename of the trained output XGBoost model");
   DeclareOptionRef(fNumEpochs, "NumEpochs", "Number of training epochs");
   DeclareOptionRef(fVerbose, "Verbose", "XGBoost verbosity during training");
   DeclareOptionRef(fContinueTraining, "ContinueTraining", "Load weights from previous training");
   DeclareOptionRef(fSaveBestOnly, "SaveBestOnly", "Store only weights with smallest validation loss");
   DeclareOptionRef(fTriesEarlyStopping, "TriesEarlyStopping", "Number of epochs with no improvement in validation loss after which training will be stopped. The default or a negative number deactivates this option.");
   DeclareOptionRef(fLearningRateSchedule, "LearningRateSchedule", "Set new learning rate during training at specific epochs, e.g., \"50,0.01;70,0.005\"");
}

void MethodPyXGBoost::ProcessOptions() {
   // Set default filename for trained model if option is not used
   if (fFilenameTrainedModel.IsNull()) {
      fFilenameTrainedModel = GetWeightFileDir() + "/TrainedModel_" + GetName() + ".model";
   }
   // Setup model, either the initial model from `fFilenameModel` or
   // the trained model from `fFilenameTrainedModel`
   if (fContinueTraining) Log() << kINFO << "Continue training with trained model" << Endl;
   SetupXGBoostModel(fContinueTraining);
}

void MethodPyXGBoost::SetupXGBoostModel(bool loadTrainedModel) {
   /*
    * Load XGBoost model from file
    */

   // Load initial model or already trained model
   TString filenameLoadModel;
   if (loadTrainedModel) {
      filenameLoadModel = fFilenameTrainedModel;
   }
   else {
      filenameLoadModel = fFilenameModel;
   }
   PyRunString("model = xgb.Booster('"+filenameLoadModel+"')",
               "Failed to load XGBoost model from file: "+filenameLoadModel);
   Log() << kINFO << "Load model from file: " << filenameLoadModel << Endl;

   /*
    * Init variables and weights
    */

   // Get variables, classes and target numbers
   fNVars = GetNVariables();
   if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) fNOutputs = DataInfo().GetNClasses();
   else if (GetAnalysisType() == Types::kRegression) fNOutputs = DataInfo().GetNTargets();
   else Log() << kFATAL << "Selected analysis type is not implemented" << Endl;

   // Init evaluation (needed for getMvaValue)
   fVals = new float[fNVars]; // holds values used for classification and regression
   npy_intp dimsVals[2] = {(npy_intp)1, (npy_intp)fNVars};
   PyArrayObject* pVals = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsVals, NPY_FLOAT, (void*)fVals);
   PyDict_SetItemString(fLocalNS, "vals", (PyObject*)pVals);

   fOutput.resize(fNOutputs); // holds classification probabilities or regression output
   npy_intp dimsOutput[2] = {(npy_intp)1, (npy_intp)fNOutputs};
   PyArrayObject* pOutput = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsOutput, NPY_FLOAT, (void*)&fOutput[0]);
   PyDict_SetItemString(fLocalNS, "output", (PyObject*)pOutput);

   // Mark the model as setup
   fModelIsSetup = true;
}

void MethodPyXGBoost::Init() {
   if (!PyIsInitialized()) {
      Log() << kFATAL << "Python is not initialized" << Endl;
   }
   _import_array(); // required to use numpy arrays

   // Import XGBoost
   PyRunString("import xgboost as xgb", "Import XGBoost failed");

   // Set flag that model is not setup
   fModelIsSetup = false;
}

void MethodPyXGBoost::Train() {
   if(!fModelIsSetup) Log() << kFATAL << "Model is not setup for training" << Endl;

   /*
    * Load training data to numpy array
    */

   UInt_t nTrainingEvents = Data()->GetNTrainingEvents();

   float* trainDataX = new float[nTrainingEvents*fNVars];
   float* trainDataY = new float[nTrainingEvents*fNOutputs];
   float* trainDataWeights = new float[nTrainingEvents];
   for (UInt_t i=0; i<nTrainingEvents; i++) {
      const TMVA::Event* e = GetTrainingEvent(i);
      // Fill variables
      for (UInt_t j=0; j<fNVars; j++) {
         trainDataX[j + i*fNVars] = e->GetValue(j);
      }
      // Fill targets
      // NOTE: For classification, convert class number in one-hot vector,
      // e.g., 1 -> [0, 1] or 0 -> [1, 0] for binary classification
      if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            trainDataY[j + i*fNOutputs] = 0;
         }
         trainDataY[e->GetClass() + i*fNOutputs] = 1;
      }
      else if (GetAnalysisType() == Types::kRegression) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            trainDataY[j + i*fNOutputs] = e->GetTarget(j);
         }
      }
      else Log() << kFATAL << "Can not fill target vector because analysis type is not known" << Endl;
      // Fill weights
      // NOTE: If no weight branch is given, this defaults to ones for all events
      trainDataWeights[i] = e->GetWeight();
   }

   npy_intp dimsTrainX[2] = {(npy_intp)nTrainingEvents, (npy_intp)fNVars};
   npy_intp dimsTrainY[2] = {(npy_intp)nTrainingEvents, (npy_intp)fNOutputs};
   npy_intp dimsTrainWeights[1] = {(npy_intp)nTrainingEvents};
   PyArrayObject* pTrainDataX = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsTrainX, NPY_FLOAT, (void*)trainDataX);
   PyArrayObject* pTrainDataY = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsTrainY, NPY_FLOAT, (void*)trainDataY);
   PyArrayObject* pTrainDataWeights = (PyArrayObject*)PyArray_SimpleNewFromData(1, dimsTrainWeights, NPY_FLOAT, (void*)trainDataWeights);
   PyDict_SetItemString(fLocalNS, "trainX", (PyObject*)pTrainDataX);
   PyDict_SetItemString(fLocalNS, "trainY", (PyObject*)pTrainDataY);
   PyDict_SetItemString(fLocalNS, "trainWeights", (PyObject*)pTrainDataWeights);

   /*
    * Load validation data to numpy array
    */

   // NOTE: In TMVA, test data is actually validation data

   UInt_t nValEvents = Data()->GetNTestEvents();

   float* valDataX = new float[nValEvents*fNVars];
   float* valDataY = new float[nValEvents*fNOutputs];
   float* valDataWeights = new float[nValEvents];
   for (UInt_t i=0; i<nValEvents; i++) {
      const TMVA::Event* e = GetTestingEvent(i);
      // Fill variables
      for (UInt_t j=0; j<fNVars; j++) {
         valDataX[j + i*fNVars] = e->GetValue(j);
      }
      // Fill targets
      if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            valDataY[j + i*fNOutputs] = 0;
         }
         valDataY[e->GetClass() + i*fNOutputs] = 1;
      }
      else if (GetAnalysisType() == Types::kRegression) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            valDataY[j + i*fNOutputs] = e->GetTarget(j);
         }
      }
      else Log() << kFATAL << "Can not fill target vector because analysis type is not known" << Endl;
      // Fill weights
      valDataWeights[i] = e->GetWeight();
   }

   npy_intp dimsValX[2] = {(npy_intp)nValEvents, (npy_intp)fNVars};
   npy_intp dimsValY[2] = {(npy_intp)nValEvents, (npy_intp)fNOutputs};
   npy_intp dimsValWeights[1] = {(npy_intp)nValEvents};
   PyArrayObject* pValDataX = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsValX, NPY_FLOAT, (void*)valDataX);
   PyArrayObject* pValDataY = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsValY, NPY_FLOAT, (void*)valDataY);
   PyArrayObject* pValDataWeights = (PyArrayObject*)PyArray_SimpleNewFromData(1, dimsValWeights, NPY_FLOAT, (void*)valDataWeights);
   PyDict_SetItemString(fLocalNS, "valX", (PyObject*)pValDataX);
   PyDict_SetItemString(fLocalNS, "valY", (PyObject*)pValDataY);
   PyDict_SetItemString(fLocalNS, "valWeights", (PyObject*)pValDataWeights);

   /*
    * Train XGBoost model
    */

   // Setup parameters

   PyObject* pNumEpochs = PyLong_FromLong(fNumEpochs);
   PyObject* pVerbose = PyLong_FromLong(fVerbose);
   PyDict_SetItemString(fLocalNS, "numRounds", pNumEpochs);
   PyDict_SetItemString(fLocalNS, "verbose", pVerbose);

   // Setup training callbacks
   PyRunString("callbacks = []");

   // Callback: Save only weights with smallest validation loss
   if (fSaveBestOnly) {
      PyRunString("callbacks.append(keras.callbacks.ModelCheckpoint('"+fFilenameTrainedModel+"', monitor='val_loss', verbose=verbose, save_best_only=True, mode='auto'))", "Failed to setup training callback: SaveBestOnly");
      Log() << kINFO << "Option SaveBestOnly: Only model weights with smallest validation loss will be stored" << Endl;
   }

   // Callback: Stop training early if no improvement in validation loss is observed
   // I didn't use callback for XGBoost, I used only earlystop in the train function
   if (fTriesEarlyStopping>=0) {
      TString tries;
      tries.Form("%i", fTriesEarlyStopping);
      PyRunString("callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience="+tries+", verbose=verbose, mode='auto'))", "Failed to setup training callback: TriesEarlyStopping");
      Log() << kINFO << "Option TriesEarlyStopping: Training will stop after " << tries << " number of epochs with no improvement of validation loss" << Endl;
   }

   // Train model
   PyRunString("bst = model.fit(param, dtrain, sample_weight=trainWeights, nb_epoch=numRounds, verbose=verbose, watchlist, early_stopping_rounds=)",
               "Failed to train model");

   /*
    * Store trained model to file (only if option 'SaveBestOnly' is NOT activated,
    * because we do not want to override the best model checkpoint)
    */

   if (!fSaveBestOnly) {
      PyRunString("model.save('"+fFilenameTrainedModel+"', overwrite=True)",
                  "Failed to save trained model: "+fFilenameTrainedModel);
      Log() << kINFO << "Trained model written to file: " << fFilenameTrainedModel << Endl;
   }

   /*
    * Clean-up
    */

   delete[] trainDataX;
   delete[] trainDataY;
   delete[] trainDataWeights;
   delete[] valDataX;
   delete[] valDataY;
   delete[] valDataWeights;
}

void MethodPyXGBoost::TestClassification() {
    MethodBase::TestClassification();
}

Double_t MethodPyXGBoost::GetMvaValue(Double_t *errLower, Double_t *errUpper) {
   // Cannot determine error
   NoErrorCalc(errLower, errUpper);

   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup) {
      // Setup the trained model
      SetupXGBoostModel(true);
   }

   // Get signal probability (called mvaValue here)
   const TMVA::Event* e = GetEvent();
   for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);
   PyRunString("for i,p in enumerate(model.predict(vals)): output[i]=p\n",
               "Failed to get predictions");

   return fOutput[TMVA::Types::kSignal];
}

std::vector<Double_t> MethodPyXGBoost::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t) {
   // Check whether the model is setup
   // NOTE: Unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup) {
      // Setup the trained model
      SetupXGBoostModel(true);
   }

   // Load data to numpy array
   Long64_t nEvents = Data()->GetNEvents();
   if (firstEvt > lastEvt || lastEvt > nEvents) lastEvt = nEvents;
   if (firstEvt < 0) firstEvt = 0;
   nEvents = lastEvt-firstEvt;

   float* data = new float[nEvents*fNVars];
   for (UInt_t i=0; i<nEvents; i++) {
      Data()->SetCurrentEvent(i);
      const TMVA::Event *e = GetEvent();
      for (UInt_t j=0; j<fNVars; j++) {
         data[j + i*fNVars] = e->GetValue(j);
      }
   }

   npy_intp dimsData[2] = {(npy_intp)nEvents, (npy_intp)fNVars};
   PyArrayObject* pDataMvaValues = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsData, NPY_FLOAT, (void*)data);
   if (pDataMvaValues==0) Log() << "Failed to load data to Python array" << Endl;

   // Get prediction for all events
   PyObject* pModel = PyDict_GetItemString(fLocalNS, "model");
   if (pModel==0) Log() << kFATAL << "Failed to get model Python object" << Endl;
   PyArrayObject* pPredictions = (PyArrayObject*) PyObject_CallMethod(pModel, (char*)"predict", (char*)"O", pDataMvaValues);
   if (pPredictions==0) Log() << kFATAL << "Failed to get predictions" << Endl;
   delete[] data;

   // Load predictions to double vector
   // NOTE: The signal probability is given at the output
   std::vector<double> mvaValues(nEvents);
   float* predictionsData = (float*) PyArray_DATA(pPredictions);
   for (UInt_t i=0; i<nEvents; i++) {
      mvaValues[i] = (double) predictionsData[i*fNOutputs + TMVA::Types::kSignal];
   }

   return mvaValues;
}

std::vector<Float_t>& MethodPyXGBoost::GetRegressionValues() {
   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup){
      // Setup the model and load weights
      SetupXGBoostModel(true);
   }

   // Get regression values
   const TMVA::Event* e = GetEvent();
   for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);
   PyRunString("for i,p in enumerate(model.predict(vals)): output[i]=p\n",
               "Failed to get predictions");

   // Use inverse transformation of targets to get final regression values
   Event * eTrans = new Event(*e);
   for (UInt_t i=0; i<fNOutputs; ++i) {
      eTrans->SetTarget(i,fOutput[i]);
   }

   const Event* eTrans2 = GetTransformationHandler().InverseTransform(eTrans);
   for (UInt_t i=0; i<fNOutputs; ++i) {
      fOutput[i] = eTrans2->GetTarget(i);
   }

   return fOutput;
}

std::vector<Float_t>& MethodPyXGBoost::GetMulticlassValues() {
   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup){
      // Setup the model and load weights
      SetupXGBoostModel(true);
   }

   // Get class probabilites
   const TMVA::Event* e = GetEvent();
   for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);
   PyRunString("for i,p in enumerate(model.predict(vals)): output[i]=p\n",
               "Failed to get predictions");

   return fOutput;
}

void MethodPyXGBoost::ReadModelFromFile() {
}

void MethodPyXGBoost::GetHelpMessage() const {
// typical length of text line:
//          "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << "XGBoost is a high-level API for the Theano and Tensorflow packages." << Endl;
   Log() << "This method wraps the training and predictions steps of the XGBoost" << Endl;
   Log() << "Python package for TMVA, so that dataloading, preprocessing and" << Endl;
   Log() << "evaluation can be done within the TMVA system. To use this XGBoost" << Endl;
   Log() << "interface, you have to generate a model with XGBoost first. Then," << Endl;
   Log() << "this model can be loaded and trained in TMVA." << Endl;
   Log() << Endl;
}
