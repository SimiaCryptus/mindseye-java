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
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.*;

@SuppressWarnings("serial")
public class FullyConnectedReferenceLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(FullyConnectedReferenceLayer.class);
  @Nullable
  public final int[] inputDims;
  @Nullable
  public final int[] outputDims;
  @Nullable
  public final Tensor weights;

  protected FullyConnectedReferenceLayer() {
    super();
    outputDims = null;
    Tensor temp_02_0001 = null;
    weights = temp_02_0001 == null ? null : temp_02_0001.addRef();
    if (null != temp_02_0001)
      temp_02_0001.freeRef();
    inputDims = null;
  }

  public FullyConnectedReferenceLayer(@Nonnull final int[] inputDims, @Nonnull final int[] outputDims) {
    this.inputDims = RefArrays.copyOf(inputDims, inputDims.length);
    this.outputDims = RefArrays.copyOf(outputDims, outputDims.length);
    final int inputs = Tensor.length(inputDims);
    final int outputs = Tensor.length(outputDims);
    Tensor temp_02_0002 = new Tensor(inputs, outputs);
    weights = temp_02_0002 == null ? null : temp_02_0002.addRef();
    if (null != temp_02_0002)
      temp_02_0002.freeRef();
    set(() -> {
      final double ratio = Math.sqrt(6. / (inputs + outputs + 1));
      final double fate = Util.R.get().nextDouble();
      return (1 - 2 * fate) * ratio;
    });
  }

  protected FullyConnectedReferenceLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> resources) {
    super(json);
    outputDims = JsonUtil.getIntArray(json.getAsJsonArray("outputDims"));
    inputDims = JsonUtil.getIntArray(json.getAsJsonArray("inputDims"));
    Tensor temp_02_0003 = Tensor.fromJson(json.get("weights"), resources);
    weights = temp_02_0003 == null ? null : temp_02_0003.addRef();
    if (null != temp_02_0003)
      temp_02_0003.freeRef();
  }

  @Nullable
  public Tensor getWeights() {
    return weights == null ? null : weights.addRef();
  }

  @Nonnull
  public FullyConnectedReferenceLayer setByCoord(@Nonnull final ToDoubleFunction<Coordinate> f) {
    weights.coordStream(true).forEach(c -> {
      RefUtil.freeRef(weights.set(c, f.applyAsDouble(c)));
    });
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedReferenceLayer setByCoord(@Nonnull final ToDoubleBiFunction<Coordinate, Coordinate> f) {
    Tensor temp_02_0008 = new Tensor(inputDims);
    temp_02_0008.coordStream(true).forEach(in -> {
      Tensor temp_02_0009 = new Tensor(outputDims);
      temp_02_0009.coordStream(true).forEach(out -> {
        weights.set(new int[] { in.getIndex(), out.getIndex() }, f.applyAsDouble(in, out));
      });
      if (null != temp_02_0009)
        temp_02_0009.freeRef();
    });
    if (null != temp_02_0008)
      temp_02_0008.freeRef();
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedReferenceLayer setWeightsLog(final double value) {
    weights.coordStream(false).forEach(c -> {
      RefUtil.freeRef(weights.set(c, (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value)));
    });
    return this.addRef();
  }

  @SuppressWarnings("unused")
  public static FullyConnectedReferenceLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new FullyConnectedReferenceLayer(json, rs);
  }

  public static @SuppressWarnings("unused") FullyConnectedReferenceLayer[] addRefs(
      FullyConnectedReferenceLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayer::addRef)
        .toArray((x) -> new FullyConnectedReferenceLayer[x]);
  }

  public static @SuppressWarnings("unused") FullyConnectedReferenceLayer[][] addRefs(
      FullyConnectedReferenceLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(FullyConnectedReferenceLayer::addRefs)
        .toArray((x) -> new FullyConnectedReferenceLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result inputResult = inObj[0].addRef();
    if (null != inObj)
      ReferenceCounting.freeRefs(inObj);
    final TensorList indata = inputResult.getData();
    @Nonnull
    int[] inputDimensions = indata.getDimensions();
    final FullyConnectedReferenceLayer fullyConnectedReferenceLayer = this.addRef();
    assert Tensor.length(inputDimensions) == Tensor.length(fullyConnectedReferenceLayer.inputDims) : RefArrays
        .toString(inputDimensions) + " == " + RefArrays.toString(fullyConnectedReferenceLayer.inputDims);
    try {
      try {
        try {
          return new Result(new TensorArray(RefIntStream.range(0, indata.length())
              .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) index -> {
                @Nullable
                final Tensor input = indata.get(index);
                @Nullable
                final Tensor output = new Tensor(outputDims);
                weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                  int[] coords = c.getCoords();
                  double prev = output.get(coords[1]);
                  double w = weights.get(c);
                  double i = input.get(coords[0]);
                  double value = prev + w * i;
                  RefUtil.freeRef(output.set(coords[1], value));
                }, input == null ? null : input.addRef(), output == null ? null : output.addRef()));
                if (null != input)
                  input.freeRef();
                return output;
              }, indata == null ? null : indata.addRef())).toArray(i -> new Tensor[i])), new Result.Accumulator() {
                {
                }

                @Override
                public void accept(DeltaSet<UUID> buffer, TensorList delta) {
                  if (!FullyConnectedReferenceLayer.this.isFrozen()) {
                    Tensor[] array = RefIntStream.range(0, indata.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                          @Nullable
                          final Tensor inputTensor = indata.get(i);
                          @Nullable
                          final Tensor deltaTensor = delta.get(i);
                          @Nonnull
                          Tensor weights = new Tensor(fullyConnectedReferenceLayer.weights.getDimensions());
                          weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                            int[] coords = c.getCoords();
                            RefUtil.freeRef(weights.set(c, inputTensor.get(coords[0]) * deltaTensor.get(coords[1])));
                          }, weights == null ? null : weights.addRef(),
                              inputTensor == null ? null : inputTensor.addRef(),
                              deltaTensor == null ? null : deltaTensor.addRef()));
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          if (null != inputTensor)
                            inputTensor.freeRef();
                          return weights;
                        }, indata == null ? null : indata.addRef(),
                            fullyConnectedReferenceLayer == null ? null : fullyConnectedReferenceLayer.addRef(),
                            delta == null ? null : delta.addRef()))
                        .toArray(i -> new Tensor[i]);
                    Tensor tensor = RefUtil.get(RefArrays.stream(Tensor.addRefs(array)).reduce((a, b) -> {
                      Tensor temp_02_0007 = a.addAndFree(b == null ? null : b.addRef());
                      if (null != b)
                        b.freeRef();
                      if (null != a)
                        a.freeRef();
                      return temp_02_0007;
                    }));
                    if (null != array)
                      ReferenceCounting.freeRefs(array);
                    Delta<UUID> temp_02_0010 = buffer.get(fullyConnectedReferenceLayer.getId(), weights.getData());
                    RefUtil.freeRef(temp_02_0010.addInPlace(tensor.getData()));
                    if (null != temp_02_0010)
                      temp_02_0010.freeRef();
                    if (null != tensor)
                      tensor.freeRef();
                  }
                  if (inputResult.isAlive()) {
                    @Nonnull
                    final TensorList tensorList = new TensorArray(RefIntStream.range(0, indata.length())
                        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                          @Nullable
                          final Tensor inputTensor = new Tensor(inputDims);
                          @Nullable
                          final Tensor deltaTensor = delta.get(i);
                          weights.coordStream(false).forEach(RefUtil.wrapInterface((Consumer<? super Coordinate>) c -> {
                            int[] coords = c.getCoords();
                            RefUtil.freeRef(inputTensor.set(coords[0],
                                inputTensor.get(coords[0]) + weights.get(c) * deltaTensor.get(coords[1])));
                          }, inputTensor == null ? null : inputTensor.addRef(),
                              deltaTensor == null ? null : deltaTensor.addRef()));
                          if (null != deltaTensor)
                            deltaTensor.freeRef();
                          return inputTensor;
                        }, delta == null ? null : delta.addRef())).toArray(i -> new Tensor[i]));
                    inputResult.accumulate(buffer == null ? null : buffer.addRef(),
                        tensorList == null ? null : tensorList);
                  }
                  if (null != delta)
                    delta.freeRef();
                  if (null != buffer)
                    buffer.freeRef();
                }

                public @SuppressWarnings("unused") void _free() {
                }
              }) {

            {
            }

            @Override
            public boolean isAlive() {
              return inputResult.isAlive() || !isFrozen();
            }

            public void _free() {
            }

          };
        } finally {
          if (null != fullyConnectedReferenceLayer)
            fullyConnectedReferenceLayer.freeRef();
        }
      } finally {
        if (null != indata)
          indata.freeRef();
      }
    } finally {
      if (null != inputResult)
        inputResult.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    json.add("outputDims", JsonUtil.getJson(outputDims));
    json.add("inputDims", JsonUtil.getJson(inputDims));
    json.add("weights", weights.getJson(resources, dataSerializer));
    return json;
  }

  @Nonnull
  public void set(@Nonnull final DoubleSupplier f) {
    RefArrays.parallelSetAll(weights.getData(), i -> f.getAsDouble());
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(@Nonnull final IntToDoubleFunction f) {
    RefUtil.freeRef(weights.set(f));
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(final double[] data) {
    RefUtil.freeRef(weights.set(data));
    return this.addRef();
  }

  @Nonnull
  public FullyConnectedReferenceLayer set(@Nonnull final Tensor data) {
    weights.set(data == null ? null : data);
    return this.addRef();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    Tensor temp_02_0012 = getWeights();
    RefList<double[]> temp_02_0011 = RefArrays.asList(temp_02_0012.getData());
    if (null != temp_02_0012)
      temp_02_0012.freeRef();
    return temp_02_0011;
  }

  public void _free() {
    if (null != weights)
      weights.freeRef();
    super._free();
  }

  public @Override @SuppressWarnings("unused") FullyConnectedReferenceLayer addRef() {
    return (FullyConnectedReferenceLayer) super.addRef();
  }

}
