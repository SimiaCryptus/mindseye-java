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
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public class BiasMetaLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasMetaLayer.class);

  public BiasMetaLayer() {
  }

  protected BiasMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static BiasMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasMetaLayer(json);
  }

  public static @SuppressWarnings("unused") BiasMetaLayer[] addRefs(BiasMetaLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasMetaLayer::addRef)
        .toArray((x) -> new BiasMetaLayer[x]);
  }

  public static @SuppressWarnings("unused") BiasMetaLayer[][] addRefs(BiasMetaLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(BiasMetaLayer::addRefs)
        .toArray((x) -> new BiasMetaLayer[x][]);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0].addRef();
    final Result in1 = inObj[1].addRef();
    ReferenceCounting.freeRefs(inObj);
    TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    final TensorList data1 = in1.getData();
    Tensor tensor1 = data1.get(0);
    if (null != data1)
      data1.freeRef();
    final Tensor[] tensors = RefIntStream.range(0, itemCnt).parallel()
        .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) dataIndex -> {
          Tensor tensor = data0.get(dataIndex);
          Tensor temp_48_0003 = tensor.mapIndex(RefUtil.wrapInterface((v, c) -> {
            return v + tensor1.get(c);
          }, tensor1 == null ? null : tensor1.addRef()));
          if (null != tensor)
            tensor.freeRef();
          return temp_48_0003;
        }, tensor1 == null ? null : tensor1.addRef(), data0 == null ? null : data0.addRef()))
        .toArray(i -> new Tensor[i]);
    if (null != tensor1)
      tensor1.freeRef();
    if (null != data0)
      data0.freeRef();
    Tensor tensor0 = tensors[0].addRef();
    try {
      try {
        try {
          try {
            return new Result(new TensorArray(Tensor.addRefs(tensors)), new Result.Accumulator() {
              {
              }

              @Override
              public void accept(DeltaSet<UUID> buffer, TensorList data) {
                if (in1.isAlive()) {
                  @Nonnull
                  TensorArray tensorArray = new TensorArray(RefIntStream.range(0, data.length())
                      .mapToObj(RefUtil.wrapInterface((IntFunction<? extends Tensor>) i -> {
                        if (i == 0)
                          return tensor0.mapCoords(RefUtil.wrapInterface((c) -> {
                            return RefIntStream.range(0, itemCnt).mapToDouble(RefUtil.wrapInterface(j -> {
                              Tensor tensor = data.get(j);
                              double temp_48_0006 = tensor.get(c);
                              if (null != tensor)
                                tensor.freeRef();
                              return temp_48_0006;
                            }, data == null ? null : data.addRef())).sum();
                          }, data == null ? null : data.addRef()));
                        else {
                          return tensor0.mapCoords(v -> 0);
                        }
                      }, data == null ? null : data.addRef(), tensor0 == null ? null : tensor0.addRef()))
                      .toArray(i -> new Tensor[i]));
                  in1.accumulate(buffer == null ? null : buffer.addRef(), tensorArray == null ? null : tensorArray);
                }
                if (in0.isAlive()) {
                  in0.accumulate(buffer == null ? null : buffer.addRef(), data == null ? null : data.addRef());
                }
                if (null != data)
                  data.freeRef();
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
                return in0.isAlive() || in1.isAlive();
              }

              public void _free() {
              }

            };
          } finally {
            if (null != tensor0)
              tensor0.freeRef();
          }
        } finally {
          if (null != tensors)
            ReferenceCounting.freeRefs(tensors);
        }
      } finally {
        if (null != in1)
          in1.freeRef();
      }
    } finally {
      if (null != in0)
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

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") BiasMetaLayer addRef() {
    return (BiasMetaLayer) super.addRef();
  }
}
