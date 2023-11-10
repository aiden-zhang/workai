#include "plugin/combine_non_max_suppression_post.h"
#include "plugin/csum.h"
#include "plugin/filter_sort.h"
#include "plugin/yolov3_box.h"
#include "plugin/non_max_suppression_gather_boxes.h"
#include "plugin/non_max_suppression.h"


namespace
{
    REGISTER_DLNNE_PLUGIN(CombineNonMaxSuppressionPluginCreator, &CombineNonMaxSuppressionPluginCreator::mNamespace);
    REGISTER_DLNNE_PLUGIN(CsumPluginCreator, &CsumPluginCreator::mNamespace);
    REGISTER_DLNNE_PLUGIN(FilterSortPluginCreator, &FilterSortPluginCreator::mNamespace);
    REGISTER_DLNNE_PLUGIN(YoloV3BoxPluginCreator_, &YoloV3BoxPluginCreator_::mNamespace);
    REGISTER_DLNNE_PLUGIN(YoloV3NMSGBPluginCreator_, &YoloV3NMSGBPluginCreator_::mNamespace);
    REGISTER_DLNNE_PLUGIN(YoloV3NMSPluginCreator_, &YoloV3NMSPluginCreator_::mNamespace);

}


