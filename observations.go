package metaverse

import (
	"math"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
)

// FlashObservationSpaces stores observation meta-data for
// each Flash game environment.
var FlashObservationSpaces = map[string]*ObservationSpace{
	"flashgames.DuskDrive-v0": &ObservationSpace{
		Width:  800,
		Height: 512,
	},
}

// ObservationSpace defines information about a Flash
// game's observation space.
type ObservationSpace struct {
	Width  int
	Height int
}

// Imager creates an Imager to convert observations from o
// into vectors.
//
// The resulting Imager will scale observations and merge
// their color channels if necessary.
// The Imager will produce the exact dimensions, even if
// doing so requires some amount of zero-padding.
func (o *ObservationSpace) Imager(c anyvec.Creator, width, height int,
	grayscale bool) *Imager {
	widthFactor := float64(o.Width) / float64(width)
	heightFactor := float64(o.Height) / float64(height)
	factor := math.Max(widthFactor, heightFactor)

	newWidth := int(essentials.Round(float64(o.Width) / factor))
	newHeight := int(essentials.Round(float64(o.Height) / factor))

	xPadding := width - newWidth
	yPadding := height - newHeight

	depth := 3
	if grayscale {
		depth = 1
	}

	return &Imager{
		creator: c,
		grayify: grayscale,
		resize: &anyconv.Resize{
			Depth:        depth,
			InputWidth:   o.Width,
			InputHeight:  o.Height,
			OutputWidth:  newWidth,
			OutputHeight: newHeight,
		},
		padding: &anyconv.Padding{
			InputWidth:    newWidth,
			InputHeight:   newHeight,
			InputDepth:    depth,
			PaddingTop:    yPadding / 2,
			PaddingBottom: (yPadding / 2) + (yPadding % 2),
			PaddingLeft:   xPadding / 2,
			PaddingRight:  (xPadding / 2) + (xPadding % 2),
		},
	}
}

// An Imager converts observations to image tensors.
type Imager struct {
	creator anyvec.Creator
	grayify bool
	resize  *anyconv.Resize
	padding *anyconv.Padding
}

// Image generates a tensor for the screen observation.
func (i *Imager) Image(obs gym.Obs) anyvec.Vector {
	rawData := obs.(gym.Uint8Obs).Uint8Obs()
	stride := 1
	if i.grayify {
		stride = 3
	}

	vectorData := make([]float64, 0, len(rawData)/stride)
	for j := 0; j < len(rawData); j += stride {
		var sum float64
		for d := 0; d < stride; d++ {
			sum += float64(rawData[j+d])
		}
		vectorData = append(vectorData, sum/float64(stride))
	}

	vector := i.creator.MakeVectorData(i.creator.MakeVectorData(vectorData))
	vector = i.padding.Apply(i.resize.Apply(anydiff.NewConst(vector), 1), 1).Output()
	anyvec.Round(vector)
	return vector
}
