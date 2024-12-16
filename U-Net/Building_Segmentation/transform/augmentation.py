import albumentations as album

def get_training_augmentation():
	train_transform = [
		album.RandomCrop(height=256, width=256, always_apply=True), # always apply는 항상 적용됨을 의미함.
		album.OneOf( # OneOf == 셋 중 하나
			[
				album.HorizontalFlip(p=1),
				album.VerticalFlip(p=1),
				album.RandomRotate90(p=1),
			],
			p=0.75,
		),
	]
	return album.Compose(train_transform)

def get_validation_augmentation():
	test_transform = [
		album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
	]
	return album.Compose(test_transform)

def to_tensor(x, **kwargs):
	return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
	_transform = []
	if preprocessing_fn:
		_transform.append(album.Lambda(image=preprocessing_fn))
	_transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

	return album.Compose(_transform)
