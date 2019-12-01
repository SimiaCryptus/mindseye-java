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
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.DoubleSupplier;
import java.util.function.IntToDoubleFunction;

@SuppressWarnings("serial")
public class BiasLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  @Nullable
  public final Tensor bias;

  protected BiasLayer() {
    super();
    bias = null;
  }

  public BiasLayer(final int... dims) {
    bias = new Tensor(dims);
  }


  protected BiasLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    bias = Tensor.fromJson(json.get("bias"), rs);
  }

  public static BiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasLayer(json, rs);
  }

  @Override
  protected void _free() {
    bias.freeRef();
    super._free();
  }

  public double[] add(@Nonnull final double[] input) {
    final double[] array = RecycleBin.DOUBLES.obtain(input.length);
    double[] bias = this.bias.getData();
    if (1 == bias.length) {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[0];
      }
    } else {
      for (int i = 0; i < array.length; i++) {
        array[i] = input[i] + bias[i];
      }
    }
    return array;
  }

  @Nonnull
  public BiasLayer addWeights(@Nonnull final DoubleSupplier f) {
    double[] bias = this.bias.getData();
    Util.add(f, bias);
    return this;
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    TensorList input;
    if (0 == inObj.length) {
      input = TensorArray.create();
    } else {
      input = inObj[0].getData();
    }
    return new Result(TensorArray.wrap(input.stream().parallel()
        .map(r -> {
          @Nonnull Tensor tensor = new Tensor(add(r.getData()), r.getDimensions());
          r.freeRef();
          return tensor;
        }).toArray(i -> new Tensor[i])),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          if (!isFrozen()) {
            final Delta<UUID> deltaBuffer = buffer.get(BiasLayer.this.getId(), bias);
            if (1 == bias.length()) {
              delta.stream().parallel().forEach(d -> {
                @Nullable final double[] array = d.getData();
                deltaBuffer.addInPlace(1 == array.length ? array : new double[]{Arrays.stream(array).sum()});
                d.freeRef();
              });
            } else {
              delta.stream().parallel().forEach(d -> {
                deltaBuffer.addInPlace(d.getData());
                d.freeRef();
              });
            }
            deltaBuffer.freeRef();
          }
          if (0 < inObj.length && inObj[0].isAlive()) {
            delta.addRef();
            inObj[0].accumulate(buffer, delta);
          }
          delta.freeRef();
        }) {

      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }


      @Override
      public boolean isAlive() {
        return 0 < inObj.length && inObj[0].isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("bias", bias.getJson(resources, dataSerializer));
    return json;
  }


  @Nonnull
  public Layer set(@Nonnull final double[] ds) {
    double[] bias = this.bias.getData();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
    return this;
  }

  @Nonnull
  public BiasLayer setWeights(@Nonnull final IntToDoubleFunction f) {
    double[] bias = this.bias.getData();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
    return this;
  }

  @Nonnull
  public BiasLayer setWeightsLog(final double value) {
    double[] bias = this.bias.getData();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
    return this;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList(bias.getData());
  }

  @Nonnull
  public BiasLayer set(@Nonnull Tensor tensor) {
    double[] bias = this.bias.getData();
    assert bias.length == tensor.length();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = tensor.get(i);
    }
    return this;
  }
}
