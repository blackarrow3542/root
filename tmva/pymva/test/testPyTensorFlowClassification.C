#include <iostream>

#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMVA/Factory.h"
#include "TMVA/Reader.h"
#include "TMVA/DataLoader.h"
#include "TMVA/PyMethodBase.h"

TString pythonSrc = "\
from keras.models import Sequential\n\
from keras.layers.core import Dense, Activation\n\
from keras import initializations\n\
from keras.optimizers import SGD\n\
\n\
model = Sequential()\n\
model.add(Dense(64, init=\"normal\", activation=\"relu\", input_dim=4))\n\
model.add(Dense(2, init=\"normal\", activation=\"softmax\"))\n\
model.compile(loss=\"categorical_crossentropy\", optimizer=SGD(lr=0.01), metrics=[\"accuracy\",])\n\
model.save(\"kerasModelClassification.h5\")\n";

TString pythonSrc = "\
import tensorflow as tf \n\
saver = tf.train.Saver() \n\
with tf.Session() as sess: \n\
    #model setup... \n\
    w = tf.Variable(tf.zeros([4,1]),name=\"weights\") \n\
    b = tf.Variable(0, name=\"bias\") \n\
    input = tf.placeholder(tf.zeros([4]),shape= , name=\"input\") \n\
    true_label = tf.placeholder(tf.zeros([1]), shape= , name=\"true_label\") \n\
    predict = tf.matmul(X, w) +b, name =\"output\" \n\
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predict,Y)) \n\
    train_op = tf.train.GradientDescentOptimizer(learning_rate, name=\"train_op\").minimize(loss)  \n\
    #actual training loop \n\
    init_op = tf.global_variables_initializer() \n\
    sess.run(init_op) \n\

    saver.save(sess, \"tensorflowModelClassification\") \n\
    sess.close() \n\";


int testPyTensorFLowClassification(){
   // Get data file
   std::cout << "Get test data..." << std::endl;
   TString fname = "./tmva_class_example.root";
   if (gSystem->AccessPathName(fname))  // file does not exist in local directory
      gSystem->Exec("curl -O http://root.cern.ch/files/tmva_class_example.root");
   TFile *input = TFile::Open(fname);

   // Build model from python file
   std::cout << "Generate keras model..." << std::endl;
   UInt_t ret;
   ret = gSystem->Exec("echo '"+pythonSrc+"' > generateTensorFLowModelClassification.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to write python code to file" << std::endl;
       return 1;
   }
   ret = gSystem->Exec("python generateTensorFLowModelClassification.py");
   if(ret!=0){
       std::cout << "[ERROR] Failed to generate model using python" << std::endl;
       return 1;
   }

   // Setup PyMVA and factory
   std::cout << "Setup TMVA..." << std::endl;
   TMVA::PyMethodBase::PyInitialize();
   TFile* outputFile = TFile::Open("ResultsTestPyTensorFLowClassification.root", "RECREATE");
   TMVA::Factory *factory = new TMVA::Factory("testPyTensorFLowClassification", outputFile,
      "!V:Silent:Color:!DrawProgressBar:AnalysisType=Classification");

   // Load data
   TMVA::DataLoader *dataloader = new TMVA::DataLoader("datasetTestPyTensorFLowClassification");

   TTree *signal = (TTree*)input->Get("TreeS");
   TTree *background = (TTree*)input->Get("TreeB");
   dataloader->AddSignalTree(signal);
   dataloader->AddBackgroundTree(background);

   dataloader->AddVariable("var1");
   dataloader->AddVariable("var2");
   dataloader->AddVariable("var3");
   dataloader->AddVariable("var4");

   dataloader->PrepareTrainingAndTestTree("",
      "SplitMode=Random:NormMode=NumEvents:!V");

   // Book and train method
   factory->BookMethod(dataloader, TMVA::Types::kPyTensorFLow, "PyTensorFLow",
      "!H:!V:VarTransform=D,G:FilenameModel=kerasModelClassification.h5:FilenameTrainedModel=trainedTensorFLowModelClassification.h5:NumEpochs=10:BatchSize=32:SaveBestOnly=false:Verbose=0");
   std::cout << "Train model..." << std::endl;
   factory->TrainAllMethods();

   // Clean-up
   delete factory;
   delete dataloader;
   delete outputFile;

   // Setup reader
   UInt_t numEvents = 100;
   std::cout << "Run reader and classify " << numEvents << " events..." << std::endl;
   TMVA::Reader *reader = new TMVA::Reader("!Color:Silent");
   Float_t vars[4];
   reader->AddVariable("var1", vars+0);
   reader->AddVariable("var2", vars+1);
   reader->AddVariable("var3", vars+2);
   reader->AddVariable("var4", vars+3);
   reader->BookMVA("PyTensorFLow", "datasetTestPyTensorFLowClassification/weights/testPyTensorFLowClassification_PyTensorFLow.weights.xml");

   // Get mean response of method on signal and background events
   signal->SetBranchAddress("var1", vars+0);
   signal->SetBranchAddress("var2", vars+1);
   signal->SetBranchAddress("var3", vars+2);
   signal->SetBranchAddress("var4", vars+3);

   background->SetBranchAddress("var1", vars+0);
   background->SetBranchAddress("var2", vars+1);
   background->SetBranchAddress("var3", vars+2);
   background->SetBranchAddress("var4", vars+3);

   Float_t meanMvaSignal = 0;
   Float_t meanMvaBackground = 0;
   for(UInt_t i=0; i<numEvents; i++){
      signal->GetEntry(i);
      meanMvaSignal += reader->EvaluateMVA("PyTensorFLow");
      background->GetEntry(i);
      meanMvaBackground += reader->EvaluateMVA("PyTensorFLow");
   }
   meanMvaSignal = meanMvaSignal/float(numEvents);
   meanMvaBackground = meanMvaBackground/float(numEvents);

   // Check whether the response is obviously better than guessing
   std::cout << "Mean MVA response on signal: " << meanMvaSignal << std::endl;
   if(meanMvaSignal < 0.6){
      std::cout << "[ERROR] Mean response on signal is " << meanMvaSignal << " (<0.6)" << std::endl;
      return 1;
   }
   std::cout << "Mean MVA response on background: " << meanMvaBackground << std::endl;
   if(meanMvaBackground > 0.4){
      std::cout << "[ERROR] Mean response on background is " << meanMvaBackground << " (>0.4)" << std::endl;
      return 1;
   }

   return 0;
}

int main(){
   int err = testPyTensorFLowClassification();
   return err;
}
