#include "combine_non_max_suppression_post.h"
#include "csum.h"
#include "filter_sort.h"
#include "non_max_suppression.h"
#include "non_max_suppression_gather_boxes.h"
#include "yolov3_box.h"

constexpr char *CombineNonMaxSuppressionPluginCreator::mNamespace;
constexpr char *CsumPluginCreator::mNamespace;
constexpr char *YoloV3BoxPluginCreator_::mNamespace;
constexpr char *YoloV3NMSGBPluginCreator_::mNamespace;
constexpr char *YoloV3NMSPluginCreator_::mNamespace;
constexpr char *FilterSortPluginCreator::mNamespace;