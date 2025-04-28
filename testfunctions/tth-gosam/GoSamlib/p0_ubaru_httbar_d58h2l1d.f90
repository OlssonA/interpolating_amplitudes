module     p0_ubaru_httbar_d58h2l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity2d58h2l1d.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd58h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd58
      complex(ki) :: brack
      acd58(1)=dotproduct(k2,qshift)
      acd58(2)=dotproduct(qshift,spval4k1)
      acd58(3)=abb58(21)
      acd58(4)=dotproduct(qshift,spval5k1)
      acd58(5)=abb58(24)
      acd58(6)=abb58(12)
      acd58(7)=dotproduct(qshift,qshift)
      acd58(8)=abb58(22)
      acd58(9)=abb58(27)
      acd58(10)=abb58(17)
      acd58(11)=abb58(10)
      acd58(12)=abb58(19)
      acd58(13)=dotproduct(qshift,spvak2l4)
      acd58(14)=abb58(13)
      acd58(15)=dotproduct(qshift,spvak2l5)
      acd58(16)=abb58(16)
      acd58(17)=dotproduct(qshift,spval3k2)
      acd58(18)=abb58(20)
      acd58(19)=dotproduct(qshift,spval3l4)
      acd58(20)=abb58(28)
      acd58(21)=dotproduct(qshift,spval3l5)
      acd58(22)=abb58(25)
      acd58(23)=dotproduct(qshift,spval5l3)
      acd58(24)=abb58(11)
      acd58(25)=abb58(9)
      acd58(26)=acd58(3)*acd58(2)
      acd58(27)=acd58(5)*acd58(4)
      acd58(26)=-acd58(6)+acd58(26)+acd58(27)
      acd58(26)=acd58(1)*acd58(26)
      acd58(27)=-acd58(8)*acd58(2)
      acd58(28)=-acd58(9)*acd58(4)
      acd58(27)=acd58(10)+acd58(28)+acd58(27)
      acd58(27)=acd58(7)*acd58(27)
      acd58(28)=-acd58(11)*acd58(2)
      acd58(29)=-acd58(12)*acd58(4)
      acd58(30)=-acd58(14)*acd58(13)
      acd58(31)=-acd58(16)*acd58(15)
      acd58(32)=-acd58(18)*acd58(17)
      acd58(33)=-acd58(20)*acd58(19)
      acd58(34)=-acd58(22)*acd58(21)
      acd58(35)=-acd58(24)*acd58(23)
      brack=acd58(25)+acd58(26)+acd58(27)+acd58(28)+acd58(29)+acd58(30)+acd58(3&
      &1)+acd58(32)+acd58(33)+acd58(34)+acd58(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd58h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd58
      complex(ki) :: brack
      acd58(1)=k2(iv1)
      acd58(2)=dotproduct(qshift,spval4k1)
      acd58(3)=abb58(21)
      acd58(4)=dotproduct(qshift,spval5k1)
      acd58(5)=abb58(24)
      acd58(6)=abb58(12)
      acd58(7)=qshift(iv1)
      acd58(8)=abb58(22)
      acd58(9)=abb58(27)
      acd58(10)=abb58(17)
      acd58(11)=spval4k1(iv1)
      acd58(12)=dotproduct(k2,qshift)
      acd58(13)=dotproduct(qshift,qshift)
      acd58(14)=abb58(10)
      acd58(15)=spval5k1(iv1)
      acd58(16)=abb58(19)
      acd58(17)=spvak2l4(iv1)
      acd58(18)=abb58(13)
      acd58(19)=spvak2l5(iv1)
      acd58(20)=abb58(16)
      acd58(21)=spval3k2(iv1)
      acd58(22)=abb58(20)
      acd58(23)=spval3l4(iv1)
      acd58(24)=abb58(28)
      acd58(25)=spval3l5(iv1)
      acd58(26)=abb58(25)
      acd58(27)=spval5l3(iv1)
      acd58(28)=abb58(11)
      acd58(29)=acd58(9)*acd58(13)
      acd58(30)=-acd58(5)*acd58(12)
      acd58(29)=acd58(30)+acd58(16)+acd58(29)
      acd58(29)=acd58(15)*acd58(29)
      acd58(30)=acd58(8)*acd58(13)
      acd58(31)=-acd58(3)*acd58(12)
      acd58(30)=acd58(31)+acd58(14)+acd58(30)
      acd58(30)=acd58(11)*acd58(30)
      acd58(31)=acd58(4)*acd58(9)
      acd58(32)=acd58(2)*acd58(8)
      acd58(31)=acd58(32)-acd58(10)+acd58(31)
      acd58(31)=acd58(7)*acd58(31)
      acd58(32)=-acd58(4)*acd58(5)
      acd58(33)=-acd58(2)*acd58(3)
      acd58(32)=acd58(33)+acd58(6)+acd58(32)
      acd58(32)=acd58(1)*acd58(32)
      acd58(33)=acd58(27)*acd58(28)
      acd58(34)=acd58(25)*acd58(26)
      acd58(35)=acd58(23)*acd58(24)
      acd58(36)=acd58(21)*acd58(22)
      acd58(37)=acd58(19)*acd58(20)
      acd58(38)=acd58(17)*acd58(18)
      brack=acd58(29)+acd58(30)+2.0_ki*acd58(31)+acd58(32)+acd58(33)+acd58(34)+&
      &acd58(35)+acd58(36)+acd58(37)+acd58(38)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd58h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd58
      complex(ki) :: brack
      acd58(1)=d(iv1,iv2)
      acd58(2)=dotproduct(qshift,spval4k1)
      acd58(3)=abb58(22)
      acd58(4)=dotproduct(qshift,spval5k1)
      acd58(5)=abb58(27)
      acd58(6)=abb58(17)
      acd58(7)=k2(iv1)
      acd58(8)=spval4k1(iv2)
      acd58(9)=abb58(21)
      acd58(10)=spval5k1(iv2)
      acd58(11)=abb58(24)
      acd58(12)=k2(iv2)
      acd58(13)=spval4k1(iv1)
      acd58(14)=spval5k1(iv1)
      acd58(15)=qshift(iv1)
      acd58(16)=qshift(iv2)
      acd58(17)=-acd58(14)*acd58(16)
      acd58(18)=-acd58(10)*acd58(15)
      acd58(17)=acd58(17)+acd58(18)
      acd58(17)=acd58(5)*acd58(17)
      acd58(18)=-acd58(13)*acd58(16)
      acd58(19)=-acd58(8)*acd58(15)
      acd58(18)=acd58(18)+acd58(19)
      acd58(18)=acd58(3)*acd58(18)
      acd58(19)=-acd58(5)*acd58(4)
      acd58(20)=-acd58(3)*acd58(2)
      acd58(19)=acd58(20)+acd58(6)+acd58(19)
      acd58(19)=acd58(1)*acd58(19)
      acd58(17)=acd58(19)+acd58(17)+acd58(18)
      acd58(18)=acd58(11)*acd58(14)
      acd58(19)=acd58(9)*acd58(13)
      acd58(18)=acd58(18)+acd58(19)
      acd58(18)=acd58(12)*acd58(18)
      acd58(19)=acd58(10)*acd58(11)
      acd58(20)=acd58(8)*acd58(9)
      acd58(19)=acd58(19)+acd58(20)
      acd58(19)=acd58(7)*acd58(19)
      brack=2.0_ki*acd58(17)+acd58(18)+acd58(19)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd58h2
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(15) :: acd58
      complex(ki) :: brack
      acd58(1)=d(iv1,iv2)
      acd58(2)=spval4k1(iv3)
      acd58(3)=abb58(22)
      acd58(4)=spval5k1(iv3)
      acd58(5)=abb58(27)
      acd58(6)=d(iv1,iv3)
      acd58(7)=spval4k1(iv2)
      acd58(8)=spval5k1(iv2)
      acd58(9)=d(iv2,iv3)
      acd58(10)=spval4k1(iv1)
      acd58(11)=spval5k1(iv1)
      acd58(12)=acd58(2)*acd58(1)
      acd58(13)=acd58(7)*acd58(6)
      acd58(14)=acd58(10)*acd58(9)
      acd58(12)=acd58(14)+acd58(13)+acd58(12)
      acd58(12)=acd58(3)*acd58(12)
      acd58(13)=acd58(4)*acd58(1)
      acd58(14)=acd58(8)*acd58(6)
      acd58(15)=acd58(11)*acd58(9)
      acd58(13)=acd58(15)+acd58(14)+acd58(13)
      acd58(13)=acd58(5)*acd58(13)
      acd58(12)=acd58(13)+acd58(12)
      brack=2.0_ki*acd58(12)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd58h2
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k4
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d58h2l1d
