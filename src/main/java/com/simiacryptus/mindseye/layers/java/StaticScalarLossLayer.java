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

@SuppressWarnings("serial")
public class StaticScalarLossLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(StaticScalarLossLayer.class);
  private double target = 0.0;

  public StaticScalarLossLayer() {
  }

  protected StaticScalarLossLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public double getTarget() {
    return target;
  }

  public void setTarget(double target) {
    this.target = target;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static StaticScalarLossLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new StaticScalarLossLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    if (1 != inObj.length) {
      RefUtil.freeRefs(inObj);
      throw new IllegalArgumentException();
    }
    //if (inObj[0].getData().length() != 1) throw new IllegalArgumentException();
    final Result in0 = inObj[0].addRef();
    RefUtil.freeRefs(inObj);
    TensorList indata = in0.getData();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          indata.addRef();
          in0.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (in0.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length()).parallel()
                .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
                  @Nullable final Tensor a = indata.get(dataIndex);
                  Tensor tensor = data.get(dataIndex);
                  final double deriv = tensor.get(0)
                      * (a.get(0) - StaticScalarLossLayer.this.getTarget() < 0 ? -1 : 1);
                  tensor.freeRef();
                  a.freeRef();
                  return new Tensor(new double[]{deriv}, 1);
                }, indata.addRef(), data.addRef()))
                .toArray(i -> new Tensor[i]));
            in0.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          indata.freeRef();
          in0.freeRef();
        }
      };
      TensorArray data = new TensorArray(RefIntStream.range(0, indata.length()).parallel()
          .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
            @Nullable final Tensor a = indata.get(dataIndex);
            final double diff = Math.abs(a.get(0) - getTarget());
            a.freeRef();
            return new Tensor(new double[]{diff}, 1);
          }, indata.addRef())).toArray(i -> new Tensor[i]));
      return new Result(data, accumulator) {
        {
          in0.addRef();
        }

        @Override
        public boolean isAlive() {
          return in0.isAlive();
        }

        @Override
        public void _free() {
          in0.freeRef();
          super._free();
        }
      };
    } finally {
      indata.freeRef();
      in0.freeRef();
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
  StaticScalarLossLayer addRef() {
    return (StaticScalarLossLayer) super.addRef();
  }
}
