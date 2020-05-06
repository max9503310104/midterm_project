#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "mbed.h"
#include "uLCD_4DGL.h"
#include "DA7212.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

DA7212 audio;
uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn button2(SW2);
InterruptIn button3(SW3);
Serial pc(USBTX, USBRX);
DigitalOut led(LED1);
//Thread t;
//EventQueue queue(8 * EVENTS_EVENT_SIZE);


int mode = 0; // 0: play, 1: select mode, 2: select song, 3: game, 4:load
int song = 0; // 0, 1, 2
int cur = 0;
int gesture_index;
int break_flag = 0;
int first = 1;

int song_table[3][50] = {
  {261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
int noteLength[3][100] = {
  {1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
char serialInBuffer[32];
int16_t waveform[kAudioTxBufferSize];



// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.7) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void playNote(int freq)
{
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
}

void select_mode()
{
    mode = 1;
}
void select_done()
{
    if (mode == 1) {
        if (cur == 0) {
            if (song < 2) {
                song++;
            } else if (song == 2) {
                song = 0;
            }
            first = 1;
            mode = 0;
        }
        if (cur == 1) {
            if (song > 0) {
                song--;
            } else if (song == 0) {
                song = 2;
            }
            first = 1;
            mode = 0;
        }
        if (cur == 2) {
            mode = 2;
        }
        if (cur == 3) {
            mode = 3;
        }
    }
    else if (mode == 2) {
        if (cur == 3) { // Load a song
            mode = 4;
        } else {  // Select a song
            song = cur;
            first = 1;
            mode = 0;
        }
    }
    break_flag = 1;
}

int main(int argc, char* argv[]) {    
  int song_i;
  
  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  //int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  button2.fall(select_mode);
  button3.fall(select_done);
  //t.start(callback(&queue, &EventQueue::dispatch_forever));

  led = 1;
  song_i = 0;
  uLCD.cls();
  while (true) {
    //cur = 0;

    if (mode == 0) {
      int length;
      if (first == 1) {
        song_i = 0;
        uLCD.cls();
        first = 0;
      }
      uLCD.locate(0, 0);
      uLCD.printf("mode = %d", mode);
      uLCD.locate(0, 1);
      uLCD.printf("song = %d", song);
      //uLCD.locate(0, 2);
      //uLCD.printf("note =       ");
      uLCD.locate(0, 2);
      uLCD.printf("note = %d    ", song_table[song][song_i]);
      uLCD.locate(0, 3);
      uLCD.printf("length = %d    ", noteLength[song][song_i]);
      
      for (int j = 0; j < 2; j++) {
        playNote(0);
      }
      for (int j = 0; j < noteLength[song][song_i] * 10; j++) {
        playNote(song_table[song][song_i]);
      }
      song_i++;
      song_i %= 50;
    }
    if (mode == 1 || mode == 2) {
      if (mode == 1) {
        uLCD.cls();
        uLCD.printf("mode = %d", mode);
        uLCD.locate(0, 1);
        uLCD.printf("  Forward");
        uLCD.locate(0, 2);
        uLCD.printf("  Backward");
        uLCD.locate(0, 3);
        uLCD.printf("  Change songs");
        uLCD.locate(0, 4);
        uLCD.printf("  Game");
      }
      if (mode == 2) {
        uLCD.cls();
        uLCD.printf("mode = %d", mode);
        uLCD.locate(0, 1);
        uLCD.printf("  Song0");
        uLCD.locate(0, 2);
        uLCD.printf("  Song1");
        uLCD.locate(0, 3);
        uLCD.printf("  Song2");
        uLCD.locate(0, 4);
        uLCD.printf("  Load songs");
      }
      cur = 0;
      playNote(0);
      while(1) {
        uLCD.locate(0, cur + 1);
        uLCD.printf("->");
        //uLCD.locate(0, 5);
        //uLCD.printf("cur = %d", gesture_index);

        // Attempt to read new data from the accelerometer
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        // If there was no new data,
        // don't try to clear the buffer again and wait until next time
        if (!got_data) {
          should_clear_buffer = false;
          continue;
        }

        // Run inference, and report any error
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed on index: %d\n", begin_index);
          continue;
        }

        // Analyze the results to obtain a prediction
        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data
        should_clear_buffer = gesture_index < label_num;

        // Produce an output
        /*if (gesture_index < label_num) {
          error_reporter->Report(config.output_message[gesture_index]);
        }*/

        if (gesture_index == 0) {
          uLCD.locate(0, cur + 1);
          uLCD.printf("  ");
          cur++;
          cur = cur % 4;
        }
        if (gesture_index == 1) {
          uLCD.locate(0, cur + 1);
          uLCD.printf("  ");
          if (cur > 0) {
            cur--;
          }
          else if (cur == 0) {
            cur = 3;
          }
        }
        if (break_flag == 1) {
          break_flag = 0;
          break;
        }
      }
    }
    if (mode == 3) {
      int score = 0;
      uLCD.cls();
      uLCD.printf("mode = %d\nscore: %d/10", mode, score);
      uLCD.line(0, 105, 127, 105, 0x00FF00);
      // note
      for (int i = 0; i < 576; i++) {
        if (mode != 3) {
          break;
        }
        // music
        if (i % 12 == 1) {
          for (int j = 0; j < 3; j++) {
            playNote(0);
          }
          for (int j = 0; j < noteLength[0][i / 12] * 1; j++) {
            playNote(song_table[0][i / 12]);
          }
        }

        if (i >= 0 && i < 48) {
          uLCD.line(24, 32 + i * 2 - 4, 56, 32 + i * 2 - 4, 0x000000);
          uLCD.line(24, 32 + i * 2, 56, 32 + i * 2, 0xFFFFFF);
        }
        if (i >= 48 && i < 96) {
          uLCD.line(72, 32 + i * 2 - 96 - 4, 104, 32 + i * 2 - 96 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 96, 104, 32 + i * 2 - 96, 0xFFFFFF);
        }
        if (i >= 96 && i < 144) {
          uLCD.line(72, 32 + i * 2 - 192 - 4, 104, 32 + i * 2 - 192 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 192, 104, 32 + i * 2 - 192, 0xFFFFFF);
        }
        if (i >= 144 && i < 192) {
          uLCD.line(24, 32 + i * 2 - 288 - 4, 56, 32 + i * 2 - 288 - 4, 0x000000);
          uLCD.line(24, 32 + i * 2 - 288, 56, 32 + i * 2 - 288, 0xFFFFFF);
        }
        if (i >= 192 && i < 240) {
          uLCD.line(72, 32 + i * 2 - 384 - 4, 104, 32 + i * 2 - 384 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 384, 104, 32 + i * 2 - 384, 0xFFFFFF);
        }
        if (i >= 240 && i < 288) {
          uLCD.line(24, 32 + i * 2 - 480 - 4, 56, 32 + i * 2 - 480 - 4, 0x000000);
          uLCD.line(24, 32 + i * 2 - 480, 56, 32 + i * 2 - 480, 0xFFFFFF);
        }
        if (i >= 288 && i < 336) {
          uLCD.line(72, 32 + i * 2 - 576 - 4, 104, 32 + i * 2 - 576 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 576, 104, 32 + i * 2 - 576, 0xFFFFFF);
        }
        if (i >= 336 && i < 384) {
          uLCD.line(72, 32 + i * 2 - 672 - 4, 104, 32 + i * 2 - 672 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 672, 104, 32 + i * 2 - 672, 0xFFFFFF);
        }
        if (i >= 384 && i < 432) {
          uLCD.line(24, 32 + i * 2 - 768 - 4, 56, 32 + i * 2 - 768 - 4, 0x000000);
          uLCD.line(24, 32 + i * 2 - 768, 56, 32 + i * 2 - 768, 0xFFFFFF);
        }
        if (i >= 432 && i < 480) {
          uLCD.line(72, 32 + i * 2 - 864 - 4, 104, 32 + i * 2 - 864 - 4, 0x000000);
          uLCD.line(72, 32 + i * 2 - 864, 104, 32 + i * 2 - 864, 0xFFFFFF);
        }
        if (i >= 480 && i < 528) {
          uLCD.line(24, 32 + i * 2 - 960 - 4, 56, 32 + i * 2 - 960 - 4, 0x000000);
          uLCD.line(24, 32 + i * 2 - 960, 56, 32 + i * 2 - 960, 0x000000);
        }
        if (i % 48 == 0) {
          uLCD.line(24, 124, 104, 124, 0x000000);
          uLCD.line(24, 126, 104, 126, 0x000000);
        }

        // score
        if (i >= 43 && i < 53 || i >= 187 && i < 197 || i >= 283 && i < 293 ||
            i >= 427 && i < 437) { 
          if (gesture_index == 1) {
            score++;
            gesture_index = 2;
          }
        } else 
        if (i >= 91 && i < 101 || i >= 139 && i < 149 || i >= 235 && i < 245 ||
            i >= 331 && i < 341 || i >= 379 && i < 389 || i >= 475 && i < 485) {
          if (gesture_index == 0) {
            score++;
            gesture_index = 2;
          }
        } else {
          gesture_index = 2;
        }
        uLCD.locate(6, 1);
        uLCD.printf("%2d",score);
        // gesture
        got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                    input_length, should_clear_buffer);

        if (!got_data) {
          should_clear_buffer = false;
          continue;
        }
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
          error_reporter->Report("Invoke failed on index: %d\n", begin_index);
          continue;
        }
        gesture_index = PredictGesture(interpreter->output(0)->data.f);
        should_clear_buffer = gesture_index < label_num;

      }

    }
    if (mode == 4) {
      int i = 0;
      int serialCount = 0;

      led = 0;
      pc.getc();
      while(i < 100)
      {
        if(pc.readable())
        {
          serialInBuffer[serialCount] = pc.getc();
          serialCount++;
          if(serialCount == 3 && i < 50)
          {
            serialInBuffer[serialCount] = '\0';
            song_table[song][i] = (int) atoi(serialInBuffer);
            if (song_table[song][i] == 256) {
              song_table[song][i] = 0;
            }
            serialCount = 0;
            i++;
          }
          if(serialCount == 3 && i >= 50)
          {
            serialInBuffer[serialCount] = '\0';
            noteLength[song][i - 50] = (int) atoi(serialInBuffer);
            noteLength[song][i - 50] -= 256;
            serialCount = 0;
            i++;
          }
        }
      }
      led = 1;
      first = 1;
      mode = 0;
    }
  }
}