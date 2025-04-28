module     p0_ubaru_httbar_d59h10l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d59h10l1d.f90
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
      use p0_ubaru_httbar_abbrevd59h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(42) :: acd59
      complex(ki) :: brack
      acd59(1)=dotproduct(k2,qshift)
      acd59(2)=dotproduct(qshift,qshift)
      acd59(3)=abb59(37)
      acd59(4)=dotproduct(qshift,spvak2k1)
      acd59(5)=abb59(18)
      acd59(6)=abb59(11)
      acd59(7)=dotproduct(l4,qshift)
      acd59(8)=abb59(42)
      acd59(9)=dotproduct(l5,qshift)
      acd59(10)=abb59(39)
      acd59(11)=abb59(19)
      acd59(12)=dotproduct(qshift,spvak2l5)
      acd59(13)=abb59(12)
      acd59(14)=abb59(13)
      acd59(15)=abb59(17)
      acd59(16)=dotproduct(qshift,spval4k1)
      acd59(17)=abb59(15)
      acd59(18)=abb59(10)
      acd59(19)=dotproduct(qshift,spvak2l4)
      acd59(20)=abb59(14)
      acd59(21)=abb59(21)
      acd59(22)=dotproduct(qshift,spval4k2)
      acd59(23)=abb59(20)
      acd59(24)=dotproduct(qshift,spval4l3)
      acd59(25)=abb59(22)
      acd59(26)=dotproduct(qshift,spval4l5)
      acd59(27)=abb59(26)
      acd59(28)=dotproduct(qshift,spval5k2)
      acd59(29)=abb59(16)
      acd59(30)=abb59(23)
      acd59(31)=-acd59(3)*acd59(1)
      acd59(32)=-acd59(11)*acd59(4)
      acd59(33)=-acd59(13)*acd59(12)
      acd59(31)=acd59(14)+acd59(33)+acd59(32)+acd59(31)
      acd59(31)=acd59(2)*acd59(31)
      acd59(32)=acd59(5)*acd59(1)
      acd59(32)=-acd59(15)+acd59(32)
      acd59(32)=acd59(4)*acd59(32)
      acd59(33)=acd59(17)*acd59(12)
      acd59(33)=-acd59(21)+acd59(33)
      acd59(33)=acd59(16)*acd59(33)
      acd59(34)=-acd59(6)*acd59(1)
      acd59(35)=-acd59(8)*acd59(7)
      acd59(36)=-acd59(10)*acd59(9)
      acd59(37)=-acd59(18)*acd59(12)
      acd59(38)=-acd59(20)*acd59(19)
      acd59(39)=-acd59(23)*acd59(22)
      acd59(40)=-acd59(25)*acd59(24)
      acd59(41)=-acd59(27)*acd59(26)
      acd59(42)=-acd59(29)*acd59(28)
      brack=acd59(30)+acd59(31)+acd59(32)+acd59(33)+acd59(34)+acd59(35)+acd59(3&
      &6)+acd59(37)+acd59(38)+acd59(39)+acd59(40)+acd59(41)+acd59(42)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd59h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(47) :: acd59
      complex(ki) :: brack
      acd59(1)=k2(iv1)
      acd59(2)=dotproduct(qshift,qshift)
      acd59(3)=abb59(37)
      acd59(4)=dotproduct(qshift,spvak2k1)
      acd59(5)=abb59(18)
      acd59(6)=abb59(11)
      acd59(7)=l4(iv1)
      acd59(8)=abb59(42)
      acd59(9)=l5(iv1)
      acd59(10)=abb59(39)
      acd59(11)=qshift(iv1)
      acd59(12)=dotproduct(k2,qshift)
      acd59(13)=abb59(19)
      acd59(14)=dotproduct(qshift,spvak2l5)
      acd59(15)=abb59(12)
      acd59(16)=abb59(13)
      acd59(17)=spvak2k1(iv1)
      acd59(18)=abb59(17)
      acd59(19)=spvak2l5(iv1)
      acd59(20)=dotproduct(qshift,spval4k1)
      acd59(21)=abb59(15)
      acd59(22)=abb59(10)
      acd59(23)=spvak2l4(iv1)
      acd59(24)=abb59(14)
      acd59(25)=spval4k1(iv1)
      acd59(26)=abb59(21)
      acd59(27)=spval4k2(iv1)
      acd59(28)=abb59(20)
      acd59(29)=spval4l3(iv1)
      acd59(30)=abb59(22)
      acd59(31)=spval4l5(iv1)
      acd59(32)=abb59(26)
      acd59(33)=spval5k2(iv1)
      acd59(34)=abb59(16)
      acd59(35)=-acd59(14)*acd59(15)
      acd59(36)=-acd59(4)*acd59(13)
      acd59(37)=-acd59(3)*acd59(12)
      acd59(35)=acd59(37)+acd59(36)+acd59(16)+acd59(35)
      acd59(35)=acd59(11)*acd59(35)
      acd59(36)=-acd59(19)*acd59(15)
      acd59(37)=-acd59(17)*acd59(13)
      acd59(36)=acd59(36)+acd59(37)
      acd59(36)=acd59(2)*acd59(36)
      acd59(37)=acd59(4)*acd59(5)
      acd59(38)=-acd59(2)*acd59(3)
      acd59(37)=acd59(38)-acd59(6)+acd59(37)
      acd59(37)=acd59(1)*acd59(37)
      acd59(38)=acd59(14)*acd59(21)
      acd59(38)=acd59(38)-acd59(26)
      acd59(38)=acd59(25)*acd59(38)
      acd59(39)=-acd59(33)*acd59(34)
      acd59(40)=-acd59(31)*acd59(32)
      acd59(41)=-acd59(29)*acd59(30)
      acd59(42)=-acd59(27)*acd59(28)
      acd59(43)=-acd59(23)*acd59(24)
      acd59(44)=-acd59(9)*acd59(10)
      acd59(45)=-acd59(7)*acd59(8)
      acd59(46)=acd59(21)*acd59(20)
      acd59(46)=-acd59(22)+acd59(46)
      acd59(46)=acd59(19)*acd59(46)
      acd59(47)=acd59(5)*acd59(12)
      acd59(47)=-acd59(18)+acd59(47)
      acd59(47)=acd59(17)*acd59(47)
      brack=2.0_ki*acd59(35)+acd59(36)+acd59(37)+acd59(38)+acd59(39)+acd59(40)+&
      &acd59(41)+acd59(42)+acd59(43)+acd59(44)+acd59(45)+acd59(46)+acd59(47)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd59h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(25) :: acd59
      complex(ki) :: brack
      acd59(1)=d(iv1,iv2)
      acd59(2)=dotproduct(k2,qshift)
      acd59(3)=abb59(37)
      acd59(4)=dotproduct(qshift,spvak2k1)
      acd59(5)=abb59(19)
      acd59(6)=dotproduct(qshift,spvak2l5)
      acd59(7)=abb59(12)
      acd59(8)=abb59(13)
      acd59(9)=k2(iv1)
      acd59(10)=qshift(iv2)
      acd59(11)=spvak2k1(iv2)
      acd59(12)=abb59(18)
      acd59(13)=k2(iv2)
      acd59(14)=qshift(iv1)
      acd59(15)=spvak2k1(iv1)
      acd59(16)=spvak2l5(iv2)
      acd59(17)=spvak2l5(iv1)
      acd59(18)=spval4k1(iv2)
      acd59(19)=abb59(15)
      acd59(20)=spval4k1(iv1)
      acd59(21)=-acd59(7)*acd59(6)
      acd59(22)=-acd59(5)*acd59(4)
      acd59(23)=-acd59(3)*acd59(2)
      acd59(21)=acd59(23)+acd59(22)+acd59(8)+acd59(21)
      acd59(21)=acd59(1)*acd59(21)
      acd59(22)=-acd59(14)*acd59(16)
      acd59(23)=-acd59(10)*acd59(17)
      acd59(22)=acd59(22)+acd59(23)
      acd59(22)=acd59(7)*acd59(22)
      acd59(23)=-acd59(14)*acd59(11)
      acd59(24)=-acd59(10)*acd59(15)
      acd59(23)=acd59(23)+acd59(24)
      acd59(23)=acd59(5)*acd59(23)
      acd59(24)=-acd59(14)*acd59(13)
      acd59(25)=-acd59(10)*acd59(9)
      acd59(24)=acd59(24)+acd59(25)
      acd59(24)=acd59(3)*acd59(24)
      acd59(21)=acd59(22)+acd59(23)+acd59(24)+acd59(21)
      acd59(22)=acd59(17)*acd59(18)
      acd59(23)=acd59(16)*acd59(20)
      acd59(22)=acd59(22)+acd59(23)
      acd59(22)=acd59(19)*acd59(22)
      acd59(23)=acd59(13)*acd59(15)
      acd59(24)=acd59(9)*acd59(11)
      acd59(23)=acd59(24)+acd59(23)
      acd59(23)=acd59(12)*acd59(23)
      brack=2.0_ki*acd59(21)+acd59(22)+acd59(23)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd59h10
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd59
      complex(ki) :: brack
      acd59(1)=d(iv1,iv2)
      acd59(2)=k2(iv3)
      acd59(3)=abb59(37)
      acd59(4)=spvak2k1(iv3)
      acd59(5)=abb59(19)
      acd59(6)=spvak2l5(iv3)
      acd59(7)=abb59(12)
      acd59(8)=d(iv1,iv3)
      acd59(9)=k2(iv2)
      acd59(10)=spvak2k1(iv2)
      acd59(11)=spvak2l5(iv2)
      acd59(12)=d(iv2,iv3)
      acd59(13)=k2(iv1)
      acd59(14)=spvak2k1(iv1)
      acd59(15)=spvak2l5(iv1)
      acd59(16)=-acd59(2)*acd59(3)
      acd59(17)=-acd59(4)*acd59(5)
      acd59(18)=-acd59(6)*acd59(7)
      acd59(16)=acd59(18)+acd59(16)+acd59(17)
      acd59(16)=acd59(1)*acd59(16)
      acd59(17)=-acd59(9)*acd59(3)
      acd59(18)=-acd59(10)*acd59(5)
      acd59(19)=-acd59(11)*acd59(7)
      acd59(17)=acd59(19)+acd59(18)+acd59(17)
      acd59(17)=acd59(8)*acd59(17)
      acd59(18)=-acd59(13)*acd59(3)
      acd59(19)=-acd59(14)*acd59(5)
      acd59(20)=-acd59(15)*acd59(7)
      acd59(18)=acd59(20)+acd59(19)+acd59(18)
      acd59(18)=acd59(12)*acd59(18)
      acd59(16)=acd59(18)+acd59(17)+acd59(16)
      brack=2.0_ki*acd59(16)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd59h10
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
      qshift = k5
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
end module     p0_ubaru_httbar_d59h10l1d
