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
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.Util;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;
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
    Tensor temp_06_0002 = new Tensor(dims);
    bias = temp_06_0002.addRef();
    temp_06_0002.freeRef();
  }

  protected BiasLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json);
    Tensor temp_06_0003 = Tensor.fromJson(json.get("bias"), rs);
    bias = temp_06_0003 == null ? null : temp_06_0003.addRef();
    if (null != temp_06_0003)
      temp_06_0003.freeRef();
  }

  public void setWeights(@Nonnull IntToDoubleFunction f) {
    assert this.bias != null;
    double[] bias = this.bias.getData();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = f.applyAsDouble(i);
    }
  }

  public void setWeightsLog(double value) {
    assert this.bias != null;
    double[] bias = this.bias.getData();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (FastRandom.INSTANCE.random() - 0.5) * Math.pow(10, value);
    }
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static BiasLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasLayer(json, rs);
  }


  public double[] add(@Nonnull final double[] input) {
    final double[] array = RecycleBin.DOUBLES.obtain(input.length);
    assert this.bias != null;
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

  public void addWeights(@Nonnull DoubleSupplier f) {
    assert this.bias != null;
    double[] bias = this.bias.getData();
    Util.add(f, bias);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList input = first(RefUtil.addRefs(inObj));
    final BiasLayer biasLayer = BiasLayer.this.addRef();
    try {
      return new Result(new TensorArray(input.stream().parallel().map(r -> {
        Tensor temp_06_0006 = new Tensor(add(r.getData()), r.getDimensions());
        r.freeRef();
        return temp_06_0006;
      }).toArray(Tensor[]::new)), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
          assert bias != null;
          bias.addRef();
          biasLayer.addRef();
        }

        @Override
        public void accept(@Nonnull DeltaSet<UUID> buffer, @Nonnull TensorList delta) {
          if (!BiasLayer.this.isFrozen()) {
            final Delta<UUID> deltaBuffer = buffer.get(biasLayer.getId(), bias == null ? null : bias.addRef());
            assert bias != null;
            if (1 == bias.length()) {
              delta.stream().parallel().forEach(RefUtil.wrapInterface((Consumer<? super Tensor>) d -> {
                @Nullable final double[] array = d.getData();
                d.freeRef();
                assert deltaBuffer != null;
                final double[] data = 1 == array.length ? array : new double[]{RefArrays.stream(array).sum()};
                deltaBuffer.addInPlace(data);
              }, deltaBuffer));
            } else {
              delta.stream().parallel().forEach(RefUtil.wrapInterface((Consumer<? super Tensor>) d -> {
                assert deltaBuffer != null;
                deltaBuffer.addInPlace(d.getData());
                d.freeRef();
              }, deltaBuffer));
            }
          }
          if (0 < inObj.length && inObj[0].isAlive()) {
            inObj[0].accumulate(buffer.addRef(), delta.addRef());
          }
          delta.freeRef();
          buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(inObj);
          assert bias != null;
          bias.freeRef();
          biasLayer.freeRef();
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          return 0 < inObj.length && inObj[0].isAlive() || !isFrozen();
        }

        public void _free() {
          RefUtil.freeRef(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(inObj);
      biasLayer.freeRef();
      input.freeRef();
    }
  }

  @NotNull
  public TensorList first(@Nonnull Result[] inObj) {
    try {
      if (0 == inObj.length) {
        return new TensorArray();
      } else {
        return inObj[0].getData();
      }
    } finally {
      RefUtil.freeRef(inObj);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, @Nonnull DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    assert bias != null;
    json.add("bias", bias.getJson(resources, dataSerializer));
    return json;
  }

  public void set(@Nonnull double[] ds) {
    assert this.bias != null;
    double[] bias = this.bias.getData();
    for (int i = 0; i < ds.length; i++) {
      bias[i] = ds[i];
    }
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    assert bias != null;
    return RefArrays.asList(bias.getData());
  }

  public void set(@Nonnull Tensor tensor) {
    assert this.bias != null;
    double[] bias = this.bias.getData();
    assert bias.length == tensor.length();
    for (int i = 0; i < bias.length; i++) {
      bias[i] = tensor.get(i);
    }
    tensor.freeRef();
  }

  public void _free() {
    if (null != bias)
      bias.freeRef();
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  BiasLayer addRef() {
    return (BiasLayer) super.addRef();
  }
}
