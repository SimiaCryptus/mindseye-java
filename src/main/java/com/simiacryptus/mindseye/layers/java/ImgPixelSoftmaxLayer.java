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
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

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

  @SuppressWarnings("unused")
  public static ImgPixelSoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSoftmaxLayer(json);
  }

  public static @SuppressWarnings("unused") ImgPixelSoftmaxLayer[] addRefs(ImgPixelSoftmaxLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSoftmaxLayer::addRef)
        .toArray((x) -> new ImgPixelSoftmaxLayer[x]);
  }

  public static @SuppressWarnings("unused") ImgPixelSoftmaxLayer[][] addRefs(ImgPixelSoftmaxLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ImgPixelSoftmaxLayer::addRefs)
        .toArray((x) -> new ImgPixelSoftmaxLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 1 == inObj.length;
    Result temp_52_0011 = eval(inObj[0].addRef());
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
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
      Tensor temp_52_0002 = temp_52_0013.setByCoord(RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return inputTensor.get(coords[0], coords[1], band);
        }, inputTensor == null ? null : inputTensor.addRef())).max().getAsDouble();
      }, inputTensor == null ? null : inputTensor.addRef()));
      if (null != temp_52_0013)
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
          Tensor temp_52_0003 = temp_52_0014.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return Math.exp(inputTensor.get(c) - maxTensor.get(coords[0], coords[1], 0));
          }, maxTensor == null ? null : maxTensor.addRef(), inputTensor == null ? null : inputTensor.addRef()));
          if (null != temp_52_0014)
            temp_52_0014.freeRef();
          if (null != maxTensor)
            maxTensor.freeRef();
          if (null != inputTensor)
            inputTensor.freeRef();
          return temp_52_0003;
        }, inputData == null ? null : inputData.addRef(), maxima == null ? null : maxima.addRef()))
        .toArray(i -> new Tensor[i]));
    if (null != maxima)
      maxima.freeRef();
    TensorArray sums = new TensorArray(exps.stream().map(expTensor -> {
      Tensor temp_52_0015 = new Tensor(width, height, 1);
      Tensor temp_52_0005 = temp_52_0015.setByCoord(RefUtil.wrapInterface(c -> {
        return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
          int[] coords = c.getCoords();
          return expTensor.get(coords[0], coords[1], band);
        }, expTensor == null ? null : expTensor.addRef())).sum();
      }, expTensor == null ? null : expTensor.addRef()));
      if (null != temp_52_0015)
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
          Tensor temp_52_0007 = temp_52_0016.setByCoord(RefUtil.wrapInterface(c -> {
            int[] coords = c.getCoords();
            return (expTensor.get(c) / sumTensor.get(coords[0], coords[1], 0));
          }, sumTensor == null ? null : sumTensor.addRef(), expTensor == null ? null : expTensor.addRef()));
          if (null != temp_52_0016)
            temp_52_0016.freeRef();
          if (null != expTensor)
            expTensor.freeRef();
          if (null != sumTensor)
            sumTensor.freeRef();
          return temp_52_0007;
        }, sums == null ? null : sums.addRef(), exps == null ? null : exps.addRef())).toArray(i -> new Tensor[i]));
    try {
      try {
        try {
          try {
            try {
              return new Result(output, new Result.Accumulator() {
                {
                  input.addRef();
                  sums.addRef();
                  inputData.addRef();
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (input.isAlive()) {
                    TensorArray dots = new TensorArray(RefIntStream.range(0, inputData.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                          final Tensor deltaTensor = delta.get(index);
                          Tensor expTensor = exps.get(index);
                          Tensor temp_52_0017 = new Tensor(width, height, 1);
                          Tensor temp_52_0009 = temp_52_0017.setByCoord(RefUtil.wrapInterface(c -> {
                            return RefIntStream.range(0, inputBands).mapToDouble(RefUtil.wrapInterface(band -> {
                              int[] coords = c.getCoords();
                              return expTensor.get(coords[0], coords[1], band)
                                  * deltaTensor.get(coords[0], coords[1], band);
                            }, deltaTensor == null ? null : deltaTensor.addRef(),
                                expTensor == null ? null : expTensor.addRef())).sum();
                          }, deltaTensor == null ? null : deltaTensor.addRef(),
                              expTensor == null ? null : expTensor.addRef()));
                          if (null != temp_52_0017)
                            temp_52_0017.freeRef();
                          if (null != expTensor)
                            expTensor.freeRef();
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          return temp_52_0009;
                        }, delta == null ? null : delta.addRef(), exps == null ? null : exps.addRef()))
                        .toArray(i -> new Tensor[i]));

                    TensorArray passback = new TensorArray(RefIntStream.range(0, inputData.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                          final Tensor deltaTensor = delta.get(index);
                          final Tensor expTensor = exps.get(index);
                          Tensor sumTensor = sums.get(index);
                          Tensor dotTensor = dots.get(index);
                          Tensor temp_52_0018 = new Tensor(inputDims);
                          Tensor temp_52_0010 = temp_52_0018.setByCoord(RefUtil.wrapInterface(c -> {
                            int[] coords = c.getCoords();
                            double sum = sumTensor.get(coords[0], coords[1], 0);
                            double dot = dotTensor.get(coords[0], coords[1], 0);
                            double deltaValue = deltaTensor.get(c);
                            double expValue = expTensor.get(c);
                            return (sum * deltaValue - dot) * expValue / (sum * sum);
                          }, deltaTensor == null ? null : deltaTensor.addRef(),
                              expTensor == null ? null : expTensor.addRef(),
                              dotTensor == null ? null : dotTensor.addRef(),
                              sumTensor == null ? null : sumTensor.addRef()));
                          if (null != temp_52_0018)
                            temp_52_0018.freeRef();
                          if (null != dotTensor)
                            dotTensor.freeRef();
                          if (null != sumTensor)
                            sumTensor.freeRef();
                          if (null != expTensor)
                            expTensor.freeRef();
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          return temp_52_0010;
                        }, sums == null ? null : sums.addRef(), dots == null ? null : dots.addRef(),
                            delta == null ? null : delta.addRef(), exps == null ? null : exps.addRef()))
                        .toArray(i -> new Tensor[i]));

                    if (null != dots)
                      dots.freeRef();
                    input.accumulate(buffer == null ? null : buffer.addRef(),
                        passback == null ? null : passback.addRef());
                    if (null != passback)
                      passback.freeRef();
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused") void _free() {
                  input.freeRef();
                  sums.freeRef();
                  inputData.freeRef();
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
                }
              };
            } finally {
              input.freeRef();
            }
          } finally {
            if (null != output)
              output.freeRef();
          }
        } finally {
          if (null != sums)
            sums.freeRef();
        }
      } finally {
        if (null != exps)
          exps.freeRef();
      }
    } finally {
      if (null != inputData)
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") ImgPixelSoftmaxLayer addRef() {
    return (ImgPixelSoftmaxLayer) super.addRef();
  }
}
