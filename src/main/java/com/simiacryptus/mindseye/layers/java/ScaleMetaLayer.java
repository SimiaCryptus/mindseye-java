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
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class ScaleMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);

  public ScaleMetaLayer() {
  }

  protected ScaleMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static ScaleMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleMetaLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    final Result in1 = inObj[1].addRef();
    RefUtil.freeRefs(inObj);
    final TensorList data0 = in0.getData();
    final TensorList data1 = in1.getData();
    final int itemCnt = data0.length();
    final Tensor data10 = data1.get(0);
    data1.freeRef();
    final Tensor[] tensors = RefIntStream.range(0, itemCnt)
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor tensor = data0.get(dataIndex);
          Tensor tensor1 = tensor
              .mapIndex(RefUtil.wrapInterface((v, c) -> v * data10.get(c), data10.addRef()));
          tensor.freeRef();
          return tensor1;
        }, data10.addRef(), data0.addRef())).toArray(i -> new Tensor[i]);
    data0.freeRef();
    Tensor tensor0 = tensors[0].addRef();
    try {
      Result.Accumulator accumulator = new Result.Accumulator() {
        {
          data10.addRef();
          tensor0.addRef();
          in0.addRef();
          in1.addRef();
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          if (in0.isAlive()) {
            @Nonnull
            TensorArray tensorArray = new TensorArray(
                data.stream().map(RefUtil.wrapInterface((Function<? super Tensor, ? extends Tensor>) t -> {
                  Tensor temp_56_0006 = t.mapIndex(RefUtil.wrapInterface((v, c) -> {
                    return v * data10.get(c);
                  }, data10.addRef()));
                  t.freeRef();
                  return temp_56_0006;
                }, data10.addRef())).toArray(i -> new Tensor[i]));
            in0.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          if (in1.isAlive()) {
            @Nullable final Tensor passback = tensor0.mapIndex(RefUtil.wrapInterface((v, c) -> {
              return RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(i -> {
                Tensor tensor = data.get(i);
                double v1 = tensor.get(c) * tensor.get(c);
                tensor.freeRef();
                return v1;
              }, data.addRef())).sum();
            }, data.addRef()));
            @Nonnull
            TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length())
                .mapToObj(RefUtil.wrapInterface(
                    (IntFunction<? extends Tensor>) i -> {
                      return i == 0 ? passback.addRef() : passback.map(v -> 0);
                    },
                    passback.addRef()))
                .toArray(i -> new Tensor[i]));
            passback.freeRef();
            in1.accumulate(buffer == null ? null : buffer.addRef(), tensorArray);
          }
          data.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          data10.freeRef();
          tensor0.freeRef();
          in0.freeRef();
          in1.freeRef();
        }
      };
      return new Result(new TensorArray(RefUtil.addRefs(tensors)), accumulator) {
        {
          in0.addRef();
          in1.addRef();
        }

        @Override
        public boolean isAlive() {
          return in0.isAlive() || in1.isAlive();
        }

        @Override
        public void _free() {
          in0.freeRef();
          in1.freeRef();
          super._free();
        }
      };
    } finally {
      tensor0.freeRef();
      RefUtil.freeRefs(tensors);
      data10.freeRef();
      in1.freeRef();
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
  ScaleMetaLayer addRef() {
    return (ScaleMetaLayer) super.addRef();
  }
}
