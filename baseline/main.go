// Command baseline trains a neural network on a single
// Universe environment.
package main

import (
	"compress/flate"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyrl/anypg"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	gym "github.com/unixpickle/gym-socket-api/binding-go"
	"github.com/unixpickle/lazyseq"
	"github.com/unixpickle/lazyseq/lazyrnn"
	"github.com/unixpickle/metaverse"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

type Flags struct {
	GymHost string
	EnvName string
	NetFile string

	FPS        int
	ScreenSize int
	Color      bool
	PastFrames int

	NumParallel int
	BatchSize   int
	Force       bool

	Discount float64
	StepSize float64
	CGIters  int
	CGFrac   float64
}

func main() {
	f := &Flags{}
	flag.StringVar(&f.GymHost, "gym", "localhost:5001", "host for gym-socket-api")
	flag.StringVar(&f.EnvName, "env", "flashgames.DuskDrive-v0", "environment name")
	flag.StringVar(&f.NetFile, "file", "policy_out", "policy neural network file")
	flag.IntVar(&f.FPS, "fps", 5, "time-steps per second")
	flag.IntVar(&f.ScreenSize, "size", 200, "longest side-length of screen")
	flag.BoolVar(&f.Color, "color", false, "use color screen images")
	flag.IntVar(&f.PastFrames, "pastframes", 1, "number of state history frames")
	flag.IntVar(&f.NumParallel, "parallel", 4, "parallel environments")
	flag.IntVar(&f.BatchSize, "batch", 12, "rollouts per batch")
	flag.BoolVar(&f.Force, "force", false, "ignore environment errors")
	flag.Float64Var(&f.Discount, "discount", 0.95, "discount factor")
	flag.Float64Var(&f.StepSize, "step", 0.01, "KL step size")
	flag.Float64Var(&f.CGFrac, "cgfrac", 0.1, "fraction of samples for CG")
	flag.IntVar(&f.CGIters, "cgiters", 10, "CG iterations per step")
	flag.Parse()

	creator := anyvec32.CurrentCreator()
	imager := MakeImager(creator, f)

	// Setup an RNNRoller for producing rollouts.
	roller := &anyrl.RNNRoller{
		Block:       MakeNetwork(creator, imager, f),
		ActionSpace: ActionSpace(f.EnvName),

		// Compress the input frames as we store them.
		// If we used a ReferenceTape for the input, the
		// program would use way too much memory.
		MakeInputTape: func() (lazyseq.Tape, chan<- *anyseq.Batch) {
			return lazyseq.CompressedUint8Tape(flate.DefaultCompression)
		},
	}

	// Setup Trust Region Policy Optimization for training.
	trpo := &anypg.TRPO{
		NaturalPG: anypg.NaturalPG{
			Policy:      roller.Block,
			Params:      anynet.AllParameters(roller.Block),
			ActionSpace: roller.ActionSpace.(anypg.NaturalActionSpace),

			// Speed things up a bit.
			Iters: f.CGIters,
			Reduce: (&anyrl.FracReducer{
				Frac:          f.CGFrac,
				MakeInputTape: roller.MakeInputTape,
			}).Reduce,

			ApplyPolicy: func(seq lazyseq.Rereader, b anyrnn.Block) lazyseq.Rereader {
				out := lazyrnn.FixedHSM(30, false, seq, b)
				return lazyseq.Lazify(lazyseq.Unlazify(out))
			},
			ActionJudger: &anypg.QJudger{Discount: f.Discount},
		},
		TargetKL: f.StepSize,
		LogLineSearch: func(kl, improvement anyvec.Numeric) {
			log.Printf("line search: kl=%f improvement=%f", kl, improvement)
		},
	}

	var saveLock sync.Mutex

	go func() {
		for batchIdx := 0; true; batchIdx++ {
			log.Println("Gathering batch of experience...")

			// Join the rollouts into one set.
			rollouts := GatherRollouts(roller, imager, f)
			r := anyrl.PackRolloutSets(rollouts)

			// Print the stats for the batch.
			log.Printf("batch %d: mean=%f stddev=%f", batchIdx,
				r.Rewards.Mean(), math.Sqrt(r.Rewards.Variance()))

			// Train on the rollouts.
			log.Println("Training on batch...")
			grad := trpo.Run(r)
			grad.AddToVars()

			saveLock.Lock()
			must(serializer.SaveAny(f.NetFile, roller.Block))
			saveLock.Unlock()
		}
	}()

	<-rip.NewRIP().Chan()
	saveLock.Lock()
	os.Exit(1)
}

func MakeImager(creator anyvec.Creator, f *Flags) *metaverse.Imager {
	obsInfo, ok := metaverse.FlashObservationSpaces[f.EnvName]
	if !ok {
		essentials.Die("unknown environment:", f.EnvName)
	}
	var downscale float64
	if obsInfo.Width > obsInfo.Height {
		downscale = float64(obsInfo.Width) / float64(f.ScreenSize)
	} else {
		downscale = float64(obsInfo.Height) / float64(f.ScreenSize)
	}
	newWidth := int(essentials.Round(float64(obsInfo.Width) / downscale))
	newHeight := int(essentials.Round(float64(obsInfo.Height) / downscale))
	return obsInfo.Imager(creator, newWidth, newHeight, !f.Color)
}

func MakeNetwork(creator anyvec.Creator, imager *metaverse.Imager,
	f *Flags) anyrnn.Block {
	var res anyrnn.Stack
	if err := serializer.LoadAny(f.NetFile, &res); err == nil {
		log.Println("Loaded network from file.")
		return res
	} else {
		width, height, depth := imager.OutSize()
		markup := fmt.Sprintf(`
			Input(w=%d, h=%d, d=%d)
			Linear(scale=0.01)
			Conv(w=4, h=4, n=16, sx=2, sy=2)
			Tanh
			Conv(w=4, h=4, n=32, sx=2, sy=2)
			Tanh
			FC(out=256)
			Tanh
		`, width, height, depth*(f.PastFrames+1))
		convNet, err := anyconv.FromMarkup(creator, markup)
		must(err)
		setupVisionLayers(convNet.(anynet.Net))
		actionSpace := ActionSpace(f.EnvName)
		res := anyrnn.Stack{
			anyrnn.NewMarkov(creator, f.PastFrames, width*height*depth, true),
			&anyrnn.LayerBlock{Layer: convNet},
			&anyrnn.LayerBlock{
				Layer: anynet.NewFCZero(creator, 256, actionSpace.ParamSize()),
			},
		}
		log.Println("Created new network.")
		return res
	}
}

func GatherRollouts(roller *anyrl.RNNRoller, imager *metaverse.Imager,
	f *Flags) []*anyrl.RolloutSet {
	resChan := make(chan *anyrl.RolloutSet, f.BatchSize)

	requests := make(chan struct{}, f.BatchSize)
	for i := 0; i < f.BatchSize; i++ {
		requests <- struct{}{}
	}
	close(requests)

	var wg sync.WaitGroup
	for i := 0; i < f.NumParallel; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			var env gym.Env
			var preproc *metaverse.Env
			var err error

			for _ = range requests {
				if env == nil {
					env, err = gym.Make(f.GymHost, f.EnvName)
					must(err)
					defer env.Close()

					// Universe-specific configuration.
					must(env.UniverseWrap("CropObservations", nil))
					must(env.UniverseWrap("Vision", nil))
					must(env.UniverseConfigure(map[string]interface{}{
						"remotes": 1,
						"fps":     f.FPS,
					}))

					preproc = &metaverse.Env{
						GymEnv:      env,
						Imager:      imager,
						ActionSpace: roller.ActionSpace.(*metaverse.ActionSpace),
					}
				}

				rollout, err := roller.Rollout(preproc)
				if !f.Force {
					must(err)
				} else if err != nil {
					log.Println("environment error:", err)
					env = nil
					continue
				}

				log.Printf("rollout: sub_reward=%f",
					rollout.Rewards.Mean())
				resChan <- rollout
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resChan)
	}()

	var res []*anyrl.RolloutSet
	for item := range resChan {
		res = append(res, item)
	}
	return res
}

func ActionSpace(envName string) *metaverse.ActionSpace {
	keyMask, ok := metaverse.FlashKeyMasks[envName]
	if !ok {
		essentials.Die("unknown environment:", envName)
	}
	pointerInfo := metaverse.FlashPointerInfo[envName]
	return &metaverse.ActionSpace{
		Keys:        keyMask,
		Pointer:     pointerInfo != nil,
		PointerInfo: pointerInfo,
	}
}

func setupVisionLayers(net anynet.Net) {
	for _, layer := range net {
		projectOutSolidColors(layer)
	}
}

func projectOutSolidColors(layer anynet.Layer) {
	switch layer := layer.(type) {
	case *anyconv.Conv:
		filters := layer.Filters.Vector
		inDepth := layer.InputDepth
		numFilters := layer.FilterCount
		filterSize := filters.Len() / numFilters
		for i := 0; i < numFilters; i++ {
			filter := filters.Slice(i*filterSize, (i+1)*filterSize)

			// Compute the mean for each input channel.
			negMean := anyvec.SumRows(filter, inDepth)
			negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(filterSize/inDepth)))
			anyvec.AddRepeated(filter, negMean)
		}
	case *anynet.FC:
		negMean := anyvec.SumCols(layer.Weights.Vector, layer.OutCount)
		negMean.Scale(negMean.Creator().MakeNumeric(-1 / float64(layer.InCount)))
		anyvec.AddChunks(layer.Weights.Vector, negMean)
	}
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}
