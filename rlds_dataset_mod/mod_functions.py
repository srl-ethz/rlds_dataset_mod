from abc import ABC, abstractmethod

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds


class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)
    

class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS = ["wrist_image", "hand_image"]

class EncodeActionToLatent(TfdsModFunction):
    SOURCE_MODALITY = ...

    def __init__(self, model_path, model_epoch, device='cpu'):
        from mmvaes import MMVAEPlusWrapper
        self.model = MMVAEPlusWrapper(model_path, model_epoch, device)

    @classmethod
    def mod_features(cls, features: tfds.features.FeaturesDict) -> tfds.features.FeaturesDict:
        return features

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def encode_action_to_latent(step):
            step["action"] = cls.model.encode_data(step["action"], cls.SOURCE_MODALITY)
            step["action"] = step["action"].numpy()
            return step

        return ds.map(encode_action_to_latent)

class EncodeManoParamsToLatent(EncodeActionToLatent):
    SOURCE_MODALITY = 'mano_params'

class EncodeGcAnglesToLatent(EncodeActionToLatent):
    SOURCE_MODALITY = 'gc_angles'

class EncodeSimpleGripperToLatent(EncodeActionToLatent):
    SOURCE_MODALITY = 'simple_gripper'

TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
    "encode_mano_params_to_latent": EncodeManoParamsToLatent,
    "encode_gc_angles_to_latent": EncodeGcAnglesToLatent,
    "encode_simple_gripper_to_latent": EncodeSimpleGripperToLatent,
}


