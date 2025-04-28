module     p0_ubaru_httbar_d57h13l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity13d57h13l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
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
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd57h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(64) :: acd57
      complex(ki) :: brack
      acd57(1)=dotproduct(k1,qshift)
      acd57(2)=abb57(30)
      acd57(3)=dotproduct(k2,qshift)
      acd57(4)=abb57(28)
      acd57(5)=abb57(11)
      acd57(6)=dotproduct(l3,qshift)
      acd57(7)=abb57(38)
      acd57(8)=dotproduct(qshift,qshift)
      acd57(9)=dotproduct(qshift,spvak1k2)
      acd57(10)=abb57(25)
      acd57(11)=dotproduct(qshift,spvak1l4)
      acd57(12)=abb57(31)
      acd57(13)=dotproduct(qshift,spvak1l5)
      acd57(14)=abb57(32)
      acd57(15)=abb57(10)
      acd57(16)=dotproduct(qshift,spvak2l4)
      acd57(17)=abb57(20)
      acd57(18)=dotproduct(qshift,spvak2l5)
      acd57(19)=abb57(21)
      acd57(20)=dotproduct(qshift,spval3l4)
      acd57(21)=abb57(15)
      acd57(22)=dotproduct(qshift,spval3l5)
      acd57(23)=abb57(27)
      acd57(24)=abb57(18)
      acd57(25)=abb57(16)
      acd57(26)=abb57(13)
      acd57(27)=abb57(26)
      acd57(28)=abb57(29)
      acd57(29)=abb57(19)
      acd57(30)=dotproduct(qshift,spvak1l3)
      acd57(31)=abb57(14)
      acd57(32)=abb57(24)
      acd57(33)=dotproduct(qshift,spvak2l3)
      acd57(34)=dotproduct(qshift,spval3k2)
      acd57(35)=abb57(22)
      acd57(36)=abb57(12)
      acd57(37)=abb57(23)
      acd57(38)=dotproduct(qshift,spval3k1)
      acd57(39)=abb57(17)
      acd57(40)=dotproduct(qshift,spval4k2)
      acd57(41)=abb57(52)
      acd57(42)=dotproduct(qshift,spval5k2)
      acd57(43)=abb57(49)
      acd57(44)=dotproduct(qshift,spval5l3)
      acd57(45)=abb57(36)
      acd57(46)=abb57(9)
      acd57(47)=acd57(20)*acd57(21)
      acd57(48)=acd57(18)*acd57(19)
      acd57(49)=acd57(16)*acd57(17)
      acd57(50)=acd57(22)*acd57(23)
      acd57(51)=-acd57(8)*acd57(10)
      acd57(47)=acd57(51)+acd57(50)+acd57(49)+acd57(48)-acd57(24)+acd57(47)
      acd57(47)=acd57(9)*acd57(47)
      acd57(48)=acd57(13)*acd57(14)
      acd57(49)=-acd57(11)*acd57(12)
      acd57(48)=acd57(49)+acd57(15)+acd57(48)
      acd57(48)=acd57(8)*acd57(48)
      acd57(49)=-acd57(44)*acd57(45)
      acd57(50)=-acd57(42)*acd57(43)
      acd57(51)=-acd57(40)*acd57(41)
      acd57(52)=-acd57(38)*acd57(39)
      acd57(53)=-acd57(6)*acd57(7)
      acd57(54)=-acd57(1)*acd57(2)
      acd57(55)=-acd57(34)*acd57(37)
      acd57(56)=-acd57(34)*acd57(35)
      acd57(56)=-acd57(36)+acd57(56)
      acd57(56)=acd57(33)*acd57(56)
      acd57(57)=-acd57(30)*acd57(32)
      acd57(58)=-acd57(20)*acd57(29)
      acd57(59)=-acd57(18)*acd57(28)
      acd57(60)=-acd57(16)*acd57(27)
      acd57(61)=-acd57(13)*acd57(26)
      acd57(62)=-acd57(11)*acd57(25)
      acd57(63)=acd57(3)*acd57(4)
      acd57(63)=-acd57(5)+acd57(63)
      acd57(63)=acd57(3)*acd57(63)
      acd57(64)=-acd57(14)*acd57(30)
      acd57(64)=-acd57(31)+acd57(64)
      acd57(64)=acd57(22)*acd57(64)
      brack=acd57(46)+acd57(47)+acd57(48)+acd57(49)+acd57(50)+acd57(51)+acd57(5&
      &2)+acd57(53)+acd57(54)+acd57(55)+acd57(56)+acd57(57)+acd57(58)+acd57(59)&
      &+acd57(60)+acd57(61)+acd57(62)+acd57(63)+acd57(64)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd57h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(77) :: acd57
      complex(ki) :: brack
      acd57(1)=k1(iv1)
      acd57(2)=abb57(30)
      acd57(3)=k2(iv1)
      acd57(4)=dotproduct(k2,qshift)
      acd57(5)=abb57(28)
      acd57(6)=abb57(11)
      acd57(7)=l3(iv1)
      acd57(8)=abb57(38)
      acd57(9)=qshift(iv1)
      acd57(10)=dotproduct(qshift,spvak1k2)
      acd57(11)=abb57(25)
      acd57(12)=dotproduct(qshift,spvak1l4)
      acd57(13)=abb57(31)
      acd57(14)=dotproduct(qshift,spvak1l5)
      acd57(15)=abb57(32)
      acd57(16)=abb57(10)
      acd57(17)=spvak1k2(iv1)
      acd57(18)=dotproduct(qshift,qshift)
      acd57(19)=dotproduct(qshift,spvak2l4)
      acd57(20)=abb57(20)
      acd57(21)=dotproduct(qshift,spvak2l5)
      acd57(22)=abb57(21)
      acd57(23)=dotproduct(qshift,spval3l4)
      acd57(24)=abb57(15)
      acd57(25)=dotproduct(qshift,spval3l5)
      acd57(26)=abb57(27)
      acd57(27)=abb57(18)
      acd57(28)=spvak1l4(iv1)
      acd57(29)=abb57(16)
      acd57(30)=spvak1l5(iv1)
      acd57(31)=abb57(13)
      acd57(32)=spvak2l4(iv1)
      acd57(33)=abb57(26)
      acd57(34)=spvak2l5(iv1)
      acd57(35)=abb57(29)
      acd57(36)=spval3l4(iv1)
      acd57(37)=abb57(19)
      acd57(38)=spval3l5(iv1)
      acd57(39)=dotproduct(qshift,spvak1l3)
      acd57(40)=abb57(14)
      acd57(41)=spvak1l3(iv1)
      acd57(42)=abb57(24)
      acd57(43)=spvak2l3(iv1)
      acd57(44)=dotproduct(qshift,spval3k2)
      acd57(45)=abb57(22)
      acd57(46)=abb57(12)
      acd57(47)=spval3k2(iv1)
      acd57(48)=dotproduct(qshift,spvak2l3)
      acd57(49)=abb57(23)
      acd57(50)=spval3k1(iv1)
      acd57(51)=abb57(17)
      acd57(52)=spval4k2(iv1)
      acd57(53)=abb57(52)
      acd57(54)=spval5k2(iv1)
      acd57(55)=abb57(49)
      acd57(56)=spval5l3(iv1)
      acd57(57)=abb57(36)
      acd57(58)=-acd57(24)*acd57(36)
      acd57(59)=-acd57(22)*acd57(34)
      acd57(60)=-acd57(20)*acd57(32)
      acd57(61)=-acd57(38)*acd57(26)
      acd57(62)=2.0_ki*acd57(9)
      acd57(63)=acd57(11)*acd57(62)
      acd57(58)=acd57(63)+acd57(61)+acd57(60)+acd57(58)+acd57(59)
      acd57(58)=acd57(10)*acd57(58)
      acd57(59)=-acd57(25)*acd57(26)
      acd57(60)=-acd57(24)*acd57(23)
      acd57(61)=-acd57(22)*acd57(21)
      acd57(63)=-acd57(20)*acd57(19)
      acd57(64)=acd57(18)*acd57(11)
      acd57(59)=acd57(64)+acd57(63)+acd57(61)+acd57(60)+acd57(27)+acd57(59)
      acd57(59)=acd57(17)*acd57(59)
      acd57(60)=acd57(25)*acd57(41)
      acd57(61)=acd57(38)*acd57(39)
      acd57(63)=-acd57(18)*acd57(30)
      acd57(60)=acd57(63)+acd57(60)+acd57(61)
      acd57(60)=acd57(15)*acd57(60)
      acd57(61)=acd57(13)*acd57(12)
      acd57(63)=-acd57(15)*acd57(14)
      acd57(61)=acd57(63)-acd57(16)+acd57(61)
      acd57(61)=acd57(61)*acd57(62)
      acd57(62)=acd57(45)*acd57(48)
      acd57(62)=acd57(62)+acd57(49)
      acd57(62)=acd57(47)*acd57(62)
      acd57(63)=acd57(18)*acd57(13)
      acd57(63)=acd57(63)+acd57(29)
      acd57(63)=acd57(28)*acd57(63)
      acd57(64)=acd57(56)*acd57(57)
      acd57(65)=acd57(54)*acd57(55)
      acd57(66)=acd57(52)*acd57(53)
      acd57(67)=acd57(50)*acd57(51)
      acd57(68)=acd57(7)*acd57(8)
      acd57(69)=acd57(1)*acd57(2)
      acd57(70)=acd57(45)*acd57(44)
      acd57(70)=acd57(46)+acd57(70)
      acd57(70)=acd57(43)*acd57(70)
      acd57(71)=acd57(41)*acd57(42)
      acd57(72)=acd57(36)*acd57(37)
      acd57(73)=acd57(34)*acd57(35)
      acd57(74)=acd57(32)*acd57(33)
      acd57(75)=acd57(30)*acd57(31)
      acd57(76)=acd57(4)*acd57(5)
      acd57(76)=acd57(6)-2.0_ki*acd57(76)
      acd57(76)=acd57(3)*acd57(76)
      acd57(77)=acd57(38)*acd57(40)
      brack=acd57(58)+acd57(59)+acd57(60)+acd57(61)+acd57(62)+acd57(63)+acd57(6&
      &4)+acd57(65)+acd57(66)+acd57(67)+acd57(68)+acd57(69)+acd57(70)+acd57(71)&
      &+acd57(72)+acd57(73)+acd57(74)+acd57(75)+acd57(76)+acd57(77)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd57h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(46) :: acd57
      complex(ki) :: brack
      acd57(1)=d(iv1,iv2)
      acd57(2)=dotproduct(qshift,spvak1k2)
      acd57(3)=abb57(25)
      acd57(4)=dotproduct(qshift,spvak1l4)
      acd57(5)=abb57(31)
      acd57(6)=dotproduct(qshift,spvak1l5)
      acd57(7)=abb57(32)
      acd57(8)=abb57(10)
      acd57(9)=k2(iv1)
      acd57(10)=k2(iv2)
      acd57(11)=abb57(28)
      acd57(12)=qshift(iv1)
      acd57(13)=spvak1k2(iv2)
      acd57(14)=spvak1l4(iv2)
      acd57(15)=spvak1l5(iv2)
      acd57(16)=qshift(iv2)
      acd57(17)=spvak1k2(iv1)
      acd57(18)=spvak1l4(iv1)
      acd57(19)=spvak1l5(iv1)
      acd57(20)=spvak2l4(iv2)
      acd57(21)=abb57(20)
      acd57(22)=spvak2l5(iv2)
      acd57(23)=abb57(21)
      acd57(24)=spval3l4(iv2)
      acd57(25)=abb57(15)
      acd57(26)=spval3l5(iv2)
      acd57(27)=abb57(27)
      acd57(28)=spvak2l4(iv1)
      acd57(29)=spvak2l5(iv1)
      acd57(30)=spval3l4(iv1)
      acd57(31)=spval3l5(iv1)
      acd57(32)=spvak1l3(iv2)
      acd57(33)=spvak1l3(iv1)
      acd57(34)=spvak2l3(iv1)
      acd57(35)=spval3k2(iv2)
      acd57(36)=abb57(22)
      acd57(37)=spvak2l3(iv2)
      acd57(38)=spval3k2(iv1)
      acd57(39)=acd57(26)*acd57(27)
      acd57(40)=acd57(25)*acd57(24)
      acd57(41)=acd57(23)*acd57(22)
      acd57(42)=acd57(21)*acd57(20)
      acd57(43)=2.0_ki*acd57(3)
      acd57(44)=-acd57(16)*acd57(43)
      acd57(39)=acd57(44)+acd57(42)+acd57(41)+acd57(39)+acd57(40)
      acd57(39)=acd57(17)*acd57(39)
      acd57(40)=acd57(27)*acd57(31)
      acd57(41)=acd57(25)*acd57(30)
      acd57(42)=acd57(23)*acd57(29)
      acd57(44)=acd57(21)*acd57(28)
      acd57(43)=-acd57(12)*acd57(43)
      acd57(40)=acd57(43)+acd57(44)+acd57(42)+acd57(40)+acd57(41)
      acd57(40)=acd57(13)*acd57(40)
      acd57(41)=-acd57(31)*acd57(32)
      acd57(42)=-acd57(26)*acd57(33)
      acd57(43)=2.0_ki*acd57(16)
      acd57(43)=acd57(19)*acd57(43)
      acd57(44)=2.0_ki*acd57(12)
      acd57(44)=acd57(15)*acd57(44)
      acd57(45)=2.0_ki*acd57(1)
      acd57(46)=acd57(6)*acd57(45)
      acd57(41)=acd57(46)+acd57(44)+acd57(43)+acd57(41)+acd57(42)
      acd57(41)=acd57(7)*acd57(41)
      acd57(42)=-acd57(16)*acd57(18)
      acd57(43)=-acd57(12)*acd57(14)
      acd57(42)=acd57(42)+acd57(43)
      acd57(42)=acd57(5)*acd57(42)
      acd57(43)=acd57(9)*acd57(10)*acd57(11)
      acd57(42)=acd57(43)+acd57(42)
      acd57(43)=-acd57(37)*acd57(38)
      acd57(44)=-acd57(34)*acd57(35)
      acd57(43)=acd57(43)+acd57(44)
      acd57(43)=acd57(36)*acd57(43)
      acd57(44)=-acd57(5)*acd57(4)
      acd57(46)=-acd57(3)*acd57(2)
      acd57(44)=acd57(46)+acd57(8)+acd57(44)
      acd57(44)=acd57(44)*acd57(45)
      brack=acd57(39)+acd57(40)+acd57(41)+2.0_ki*acd57(42)+acd57(43)+acd57(44)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd57h13_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd57
      complex(ki) :: brack
      acd57(1)=d(iv1,iv2)
      acd57(2)=spvak1k2(iv3)
      acd57(3)=abb57(25)
      acd57(4)=spvak1l4(iv3)
      acd57(5)=abb57(31)
      acd57(6)=spvak1l5(iv3)
      acd57(7)=abb57(32)
      acd57(8)=d(iv1,iv3)
      acd57(9)=spvak1k2(iv2)
      acd57(10)=spvak1l4(iv2)
      acd57(11)=spvak1l5(iv2)
      acd57(12)=d(iv2,iv3)
      acd57(13)=spvak1k2(iv1)
      acd57(14)=spvak1l4(iv1)
      acd57(15)=spvak1l5(iv1)
      acd57(16)=acd57(2)*acd57(3)
      acd57(17)=acd57(4)*acd57(5)
      acd57(18)=-acd57(6)*acd57(7)
      acd57(16)=acd57(18)+acd57(16)+acd57(17)
      acd57(16)=acd57(1)*acd57(16)
      acd57(17)=acd57(9)*acd57(3)
      acd57(18)=acd57(10)*acd57(5)
      acd57(19)=-acd57(11)*acd57(7)
      acd57(17)=acd57(19)+acd57(18)+acd57(17)
      acd57(17)=acd57(8)*acd57(17)
      acd57(18)=acd57(13)*acd57(3)
      acd57(19)=acd57(14)*acd57(5)
      acd57(20)=-acd57(15)*acd57(7)
      acd57(18)=acd57(20)+acd57(19)+acd57(18)
      acd57(18)=acd57(12)*acd57(18)
      acd57(16)=acd57(18)+acd57(17)+acd57(16)
      brack=2.0_ki*acd57(16)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd57h13_qp
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
      qshift = k3+k5
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
end module     p0_ubaru_httbar_d57h13l1d_qp
