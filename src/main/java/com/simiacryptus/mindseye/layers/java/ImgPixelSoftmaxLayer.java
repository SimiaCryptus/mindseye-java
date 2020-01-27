/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;
import java.util.function.ToDoubleFunction;

@SuppressWarnings("serial")
public class ImgPixelSoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSoftmaxLayer.class);

  public ImgPixelSoftmaxLayer() {
    super();
  }

  protected ImgPixelSoftmaxLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ImgPixelSoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSoftmaxLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Result temp_52_0011 = eval(inObj[0].addRef());
    RefUtil.freeRefs(inObj);
    return temp_52_0011;
  }

  @Nonnull
  public Result eval(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    final int inputBands = inputDims[2];
    final int width = inputDims[0];
    final int height = inputDims[1];
    TensorArray maxima = new TensorArray(inputData.stream().map(inputTensor -> {
      Tensor temp_52_0013 = new Tensor(width, height, 1);
      final ToDoubleFunction<Coordinate> f = RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return inputTensor.get(coords[0], coords[1], band);
        }, inputTensor == null ? null : inputTensor.addRef())).max().getAsDouble();
      }, inputTensor == null ? null : inputTensor.addRef());
      temp_52_0013.setByCoord(f);
      Tensor temp_52_0002 = temp_52_0013.addRef();
      temp_52_0013.freeRef();
      if (null != inputTensor)
        inputTensor.freeRef();
      return temp_52_0002;
    }).toArray(i -> new Tensor[i]));
    TensorArray exps = new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
          final Tensor inputTensor = inputData.get(index);
          Tensor maxTensor = maxima.get(index);
          Tensor temp_52_0014 = new Tensor(inputDims);
          temp_52_0014.setByCoord(RefUtil.wrapInterface(c -> {
                  int[] coords = c.getCoords();
                  return Math.exp(inputTensor.get(c) - maxTensor.get(coords[0], coords[1], 0));
                }, maxTensor.addRef(), inputTensor.addRef()));
          Tensor temp_52_0003 = temp_52_0014.addRef();
          temp_52_0014.freeRef();
          maxTensor.freeRef();
          inputTensor.freeRef();
          return temp_52_0003;
        }, inputData.addRef(), maxima.addRef()))
        .toArray(i -> new Tensor[i]));
    maxima.freeRef();
    TensorArray sums = new TensorArray(exps.stream().map(expTensor -> {
      Tensor temp_52_0015 = new Tensor(width, height, 1);
      final ToDoubleFunction<Coordinate> f = RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return expTensor.get(coords[0], coords[1], band);
        }, expTensor == null ? null : expTensor.addRef())).sum();
      }, expTensor == null ? null : expTensor.addRef());
      temp_52_0015.setByCoord(f);
      Tensor temp_52_0005 = temp_52_0015.addRef();
      temp_52_0015.freeRef();
      if (null != expTensor)
        expTensor.freeRef();
      return temp_52_0005;
    }).toArray(i -> new Tensor[i]));
    TensorArray output = new TensorArray(RefIntStream.range(0, inputData.length())
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
          Tensor sumTensor = sums.get(index);
          Tensor expTensor = exps.get(index);
          Tensor temp_52_0016 = new Tensor(inputDims);
          temp_52_0016.setByCoord(RefUtil.wrapInterface(c -> {
                  int[] coords = c.getCoords();
                  return (expTensor.get(c) / sumTensor.get(coords[0], coords[1], 0));
                }, sumTensor.addRef(), expTensor.addRef()));
          Tensor temp_52_0007 = temp_52_0016.addRef();
          temp_52_0016.freeRef();
          expTensor.freeRef();
          sumTensor.freeRef();
          return temp_52_0007;
        }, sums.addRef(), exps.addRef())).toArray(i -> new Tensor[i]));
    try {
      return new Result(output, new Result.Accumulator() {
        {
          input.addRef();
          sums.addRef();
          inputData.addRef();
          exps.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (input.isAlive()) {
            TensorArray dots = new TensorArray(RefIntStream.range(0, inputData.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                  final Tensor deltaTensor = delta.get(index);
                  Tensor expTensor = exps.get(index);
                  Tensor temp_52_0017 = new Tensor(width, height, 1);
                  temp_52_0017.setByCoord(RefUtil.wrapInterface(c -> {
                                      return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
                                            int[] coords = c.getCoords();
                                            return expTensor.get(coords[0], coords[1], band)
                                                * deltaTensor.get(coords[0], coords[1], band);
                                          }, deltaTensor.addRef(),
                                          expTensor.addRef())).sum();
                                    }, deltaTensor.addRef(),
                                    expTensor.addRef()));
                  Tensor temp_52_0009 = temp_52_0017.addRef();
                  temp_52_0017.freeRef();
                  expTensor.freeRef();
                  deltaTensor.freeRef();
                  return temp_52_0009;
                }, delta.addRef(), exps.addRef()))
                .toArray(i -> new Tensor[i]));

            TensorArray passback = new TensorArray(RefIntStream.range(0, inputData.length())
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                      final Tensor deltaTensor = delta.get(index);
                      final Tensor expTensor = exps.get(index);
                      Tensor sumTensor = sums.get(index);
                      Tensor dotTensor = dots.get(index);
                      Tensor temp_52_0018 = new Tensor(inputDims);
                      temp_52_0018.setByCoord(RefUtil.wrapInterface(c -> {
                                              int[] coords = c.getCoords();
                                              double sum = sumTensor.get(coords[0], coords[1], 0);
                                              double dot = dotTensor.get(coords[0], coords[1], 0);
                                              double deltaValue = deltaTensor.get(c);
                                              double expValue = expTensor.get(c);
                                              return (sum * deltaValue - dot) * expValue / (sum * sum);
                                            }, deltaTensor.addRef(),
                                            expTensor.addRef(),
                                            dotTensor.addRef(),
                                            sumTensor.addRef()));
                      Tensor temp_52_0010 = temp_52_0018.addRef();
                      temp_52_0018.freeRef();
                      dotTensor.freeRef();
                      sumTensor.freeRef();
                      expTensor.freeRef();
                      deltaTensor.freeRef();
                      return temp_52_0010;
                    }, sums.addRef(), dots.addRef(),
                    delta.addRef(), exps.addRef()))
                .toArray(i -> new Tensor[i]));

            dots.freeRef();
            input.accumulate(buffer == null ? null : buffer.addRef(),
                passback.addRef());
            passback.freeRef();
          }
          delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          input.freeRef();
          sums.freeRef();
          inputData.freeRef();
          exps.freeRef();
        }
      }) {

        {
          input.addRef();
        }

        @Override
        public boolean isAlive() {
          return input.isAlive() || !isFrozen();
        }

        public void _free() {
          input.freeRef();
          super._free();
        }
      };
    } finally {
      input.freeRef();
      sums.freeRef();
      exps.freeRef();
      inputData.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgPixelSoftmaxLayer addRef() {
    return (ImgPixelSoftmaxLayer) super.addRef();
  }
}
