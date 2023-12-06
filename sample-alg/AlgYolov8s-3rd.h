#ifndef __TS_ALG_YOLOV8S_H__
#define __TS_ALG_YOLOV8S_H__

//
// headers included
//
#include <Common.h>

//
// functions
//
extern "C"  void* algInit  (const std::string& args              );
extern "C"  bool  algStart (void* alg                            );
extern "C"  std::shared_ptr<TsJsonObject> algProc (void* alg,
    const std::shared_ptr<TsGstSample>& data                     );
extern "C"  bool  algCtrl  (void* alg, const std::string& cmd    );
extern "C"  void  algStop  (void* alg                            );
extern "C"  void  algFina  (void* alg                            );
extern "C"  bool  algSetCb (void* alg, TsPutResult cb, void* args);

#endif //__TS_ALG_YOLOV8S_H__

