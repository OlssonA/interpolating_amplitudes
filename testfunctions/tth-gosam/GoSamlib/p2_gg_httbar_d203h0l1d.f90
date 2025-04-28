module     p2_gg_httbar_d203h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d203h0l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd203h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd203
      complex(ki) :: brack
      acd203(1)=dotproduct(qshift,spvae1k2)
      acd203(2)=dotproduct(qshift,spval3e2)
      acd203(3)=dotproduct(qshift,spvae2e1)
      acd203(4)=abb203(47)
      acd203(5)=dotproduct(qshift,spval4e2)
      acd203(6)=abb203(68)
      acd203(7)=dotproduct(qshift,spval5e2)
      acd203(8)=abb203(72)
      acd203(9)=abb203(45)
      acd203(10)=abb203(56)
      acd203(11)=abb203(58)
      acd203(12)=abb203(62)
      acd203(13)=dotproduct(qshift,spvae1l3)
      acd203(14)=abb203(49)
      acd203(15)=abb203(67)
      acd203(16)=abb203(52)
      acd203(17)=abb203(64)
      acd203(18)=abb203(57)
      acd203(19)=abb203(53)
      acd203(20)=abb203(54)
      acd203(21)=abb203(63)
      acd203(22)=dotproduct(qshift,spvae2k2)
      acd203(23)=dotproduct(qshift,spval3e1)
      acd203(24)=dotproduct(qshift,spvae1e2)
      acd203(25)=dotproduct(qshift,spval4e1)
      acd203(26)=dotproduct(qshift,spval5e1)
      acd203(27)=abb203(73)
      acd203(28)=abb203(69)
      acd203(29)=abb203(59)
      acd203(30)=dotproduct(qshift,spvae2l3)
      acd203(31)=abb203(65)
      acd203(32)=abb203(66)
      acd203(33)=abb203(61)
      acd203(34)=abb203(55)
      acd203(35)=abb203(48)
      acd203(36)=abb203(71)
      acd203(37)=abb203(60)
      acd203(38)=abb203(50)
      acd203(39)=abb203(46)
      acd203(40)=-acd203(2)*acd203(4)
      acd203(41)=-acd203(7)*acd203(8)
      acd203(42)=-acd203(5)*acd203(6)
      acd203(40)=acd203(42)+acd203(41)-acd203(9)+acd203(40)
      acd203(40)=acd203(1)*acd203(40)
      acd203(41)=acd203(2)*acd203(11)
      acd203(42)=-acd203(13)*acd203(18)
      acd203(43)=-acd203(13)*acd203(16)
      acd203(43)=acd203(17)+acd203(43)
      acd203(43)=acd203(7)*acd203(43)
      acd203(44)=-acd203(13)*acd203(14)
      acd203(44)=acd203(15)+acd203(44)
      acd203(44)=acd203(5)*acd203(44)
      acd203(40)=acd203(40)+acd203(44)+acd203(43)+acd203(42)-acd203(19)+acd203(&
      &41)
      acd203(40)=acd203(3)*acd203(40)
      acd203(41)=-acd203(4)*acd203(23)
      acd203(42)=-acd203(26)*acd203(8)
      acd203(43)=-acd203(25)*acd203(6)
      acd203(41)=acd203(43)+acd203(42)+acd203(27)+acd203(41)
      acd203(41)=acd203(22)*acd203(41)
      acd203(42)=acd203(30)*acd203(33)
      acd203(43)=-acd203(30)*acd203(16)
      acd203(43)=acd203(32)+acd203(43)
      acd203(43)=acd203(26)*acd203(43)
      acd203(44)=-acd203(30)*acd203(14)
      acd203(44)=acd203(31)+acd203(44)
      acd203(44)=acd203(25)*acd203(44)
      acd203(41)=acd203(41)+acd203(44)+acd203(43)-acd203(34)+acd203(42)
      acd203(41)=acd203(24)*acd203(41)
      acd203(42)=-acd203(23)*acd203(29)
      acd203(43)=-acd203(2)*acd203(12)
      acd203(44)=-acd203(30)*acd203(38)
      acd203(45)=acd203(26)*acd203(36)
      acd203(46)=acd203(25)*acd203(35)
      acd203(47)=-acd203(13)*acd203(37)
      acd203(48)=-acd203(7)*acd203(21)
      acd203(49)=-acd203(5)*acd203(20)
      acd203(50)=-acd203(22)*acd203(28)
      acd203(51)=-acd203(1)*acd203(10)
      brack=acd203(39)+acd203(40)+acd203(41)+acd203(42)+acd203(43)+acd203(44)+a&
      &cd203(45)+acd203(46)+acd203(47)+acd203(48)+acd203(49)+acd203(50)+acd203(&
      &51)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd203h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(64) :: acd203
      complex(ki) :: brack
      acd203(1)=spvae1k2(iv1)
      acd203(2)=dotproduct(qshift,spval3e2)
      acd203(3)=dotproduct(qshift,spvae2e1)
      acd203(4)=abb203(47)
      acd203(5)=dotproduct(qshift,spval4e2)
      acd203(6)=abb203(68)
      acd203(7)=dotproduct(qshift,spval5e2)
      acd203(8)=abb203(72)
      acd203(9)=abb203(45)
      acd203(10)=abb203(56)
      acd203(11)=spval3e2(iv1)
      acd203(12)=dotproduct(qshift,spvae1k2)
      acd203(13)=abb203(58)
      acd203(14)=abb203(62)
      acd203(15)=spvae2e1(iv1)
      acd203(16)=dotproduct(qshift,spvae1l3)
      acd203(17)=abb203(49)
      acd203(18)=abb203(67)
      acd203(19)=abb203(52)
      acd203(20)=abb203(64)
      acd203(21)=abb203(57)
      acd203(22)=abb203(53)
      acd203(23)=spval4e2(iv1)
      acd203(24)=abb203(54)
      acd203(25)=spval5e2(iv1)
      acd203(26)=abb203(63)
      acd203(27)=spvae2k2(iv1)
      acd203(28)=dotproduct(qshift,spval3e1)
      acd203(29)=dotproduct(qshift,spvae1e2)
      acd203(30)=dotproduct(qshift,spval4e1)
      acd203(31)=dotproduct(qshift,spval5e1)
      acd203(32)=abb203(73)
      acd203(33)=abb203(69)
      acd203(34)=spval3e1(iv1)
      acd203(35)=dotproduct(qshift,spvae2k2)
      acd203(36)=abb203(59)
      acd203(37)=spvae1e2(iv1)
      acd203(38)=dotproduct(qshift,spvae2l3)
      acd203(39)=abb203(65)
      acd203(40)=abb203(66)
      acd203(41)=abb203(61)
      acd203(42)=abb203(55)
      acd203(43)=spval4e1(iv1)
      acd203(44)=abb203(48)
      acd203(45)=spval5e1(iv1)
      acd203(46)=abb203(71)
      acd203(47)=spvae1l3(iv1)
      acd203(48)=abb203(60)
      acd203(49)=spvae2l3(iv1)
      acd203(50)=abb203(50)
      acd203(51)=acd203(8)*acd203(7)
      acd203(52)=acd203(6)*acd203(5)
      acd203(53)=acd203(4)*acd203(2)
      acd203(51)=acd203(51)+acd203(52)+acd203(53)+acd203(9)
      acd203(52)=-acd203(1)*acd203(51)
      acd203(53)=-acd203(8)*acd203(25)
      acd203(54)=-acd203(6)*acd203(23)
      acd203(55)=-acd203(4)*acd203(11)
      acd203(53)=acd203(55)+acd203(53)+acd203(54)
      acd203(53)=acd203(12)*acd203(53)
      acd203(54)=-acd203(16)*acd203(25)
      acd203(55)=-acd203(7)*acd203(47)
      acd203(54)=acd203(54)+acd203(55)
      acd203(54)=acd203(19)*acd203(54)
      acd203(55)=-acd203(16)*acd203(23)
      acd203(56)=-acd203(5)*acd203(47)
      acd203(55)=acd203(55)+acd203(56)
      acd203(55)=acd203(17)*acd203(55)
      acd203(56)=acd203(11)*acd203(13)
      acd203(57)=-acd203(47)*acd203(21)
      acd203(58)=acd203(25)*acd203(20)
      acd203(59)=acd203(23)*acd203(18)
      acd203(52)=acd203(53)+acd203(55)+acd203(54)+acd203(59)+acd203(58)+acd203(&
      &56)+acd203(57)+acd203(52)
      acd203(52)=acd203(3)*acd203(52)
      acd203(53)=acd203(8)*acd203(31)
      acd203(54)=acd203(6)*acd203(30)
      acd203(55)=acd203(4)*acd203(28)
      acd203(53)=acd203(53)+acd203(54)+acd203(55)-acd203(32)
      acd203(54)=-acd203(27)*acd203(53)
      acd203(55)=-acd203(8)*acd203(45)
      acd203(56)=-acd203(6)*acd203(43)
      acd203(57)=-acd203(4)*acd203(34)
      acd203(55)=acd203(57)+acd203(55)+acd203(56)
      acd203(55)=acd203(35)*acd203(55)
      acd203(56)=-acd203(38)*acd203(45)
      acd203(57)=-acd203(31)*acd203(49)
      acd203(56)=acd203(56)+acd203(57)
      acd203(56)=acd203(19)*acd203(56)
      acd203(57)=-acd203(38)*acd203(43)
      acd203(58)=-acd203(30)*acd203(49)
      acd203(57)=acd203(57)+acd203(58)
      acd203(57)=acd203(17)*acd203(57)
      acd203(58)=acd203(49)*acd203(41)
      acd203(59)=acd203(45)*acd203(40)
      acd203(60)=acd203(43)*acd203(39)
      acd203(54)=acd203(55)+acd203(57)+acd203(56)+acd203(60)+acd203(58)+acd203(&
      &59)+acd203(54)
      acd203(54)=acd203(29)*acd203(54)
      acd203(51)=-acd203(12)*acd203(51)
      acd203(55)=-acd203(19)*acd203(7)
      acd203(56)=-acd203(17)*acd203(5)
      acd203(55)=acd203(56)+acd203(55)-acd203(21)
      acd203(55)=acd203(16)*acd203(55)
      acd203(56)=acd203(2)*acd203(13)
      acd203(57)=acd203(7)*acd203(20)
      acd203(58)=acd203(5)*acd203(18)
      acd203(51)=acd203(51)+acd203(58)+acd203(57)-acd203(22)+acd203(56)+acd203(&
      &55)
      acd203(51)=acd203(15)*acd203(51)
      acd203(53)=-acd203(35)*acd203(53)
      acd203(55)=-acd203(19)*acd203(31)
      acd203(56)=-acd203(17)*acd203(30)
      acd203(55)=acd203(56)+acd203(55)+acd203(41)
      acd203(55)=acd203(38)*acd203(55)
      acd203(56)=acd203(31)*acd203(40)
      acd203(57)=acd203(30)*acd203(39)
      acd203(53)=acd203(53)+acd203(57)+acd203(56)-acd203(42)+acd203(55)
      acd203(53)=acd203(37)*acd203(53)
      acd203(55)=-acd203(34)*acd203(36)
      acd203(56)=-acd203(11)*acd203(14)
      acd203(57)=-acd203(49)*acd203(50)
      acd203(58)=-acd203(47)*acd203(48)
      acd203(59)=acd203(45)*acd203(46)
      acd203(60)=acd203(43)*acd203(44)
      acd203(61)=-acd203(25)*acd203(26)
      acd203(62)=-acd203(23)*acd203(24)
      acd203(63)=-acd203(27)*acd203(33)
      acd203(64)=-acd203(1)*acd203(10)
      brack=acd203(51)+acd203(52)+acd203(53)+acd203(54)+acd203(55)+acd203(56)+a&
      &cd203(57)+acd203(58)+acd203(59)+acd203(60)+acd203(61)+acd203(62)+acd203(&
      &63)+acd203(64)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd203h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(71) :: acd203
      complex(ki) :: brack
      acd203(1)=spvae1k2(iv1)
      acd203(2)=spval3e2(iv2)
      acd203(3)=dotproduct(qshift,spvae2e1)
      acd203(4)=abb203(47)
      acd203(5)=spvae2e1(iv2)
      acd203(6)=dotproduct(qshift,spval3e2)
      acd203(7)=dotproduct(qshift,spval4e2)
      acd203(8)=abb203(68)
      acd203(9)=dotproduct(qshift,spval5e2)
      acd203(10)=abb203(72)
      acd203(11)=abb203(45)
      acd203(12)=spval4e2(iv2)
      acd203(13)=spval5e2(iv2)
      acd203(14)=spvae1k2(iv2)
      acd203(15)=spval3e2(iv1)
      acd203(16)=spvae2e1(iv1)
      acd203(17)=spval4e2(iv1)
      acd203(18)=spval5e2(iv1)
      acd203(19)=dotproduct(qshift,spvae1k2)
      acd203(20)=abb203(58)
      acd203(21)=dotproduct(qshift,spvae1l3)
      acd203(22)=abb203(49)
      acd203(23)=abb203(67)
      acd203(24)=abb203(52)
      acd203(25)=abb203(64)
      acd203(26)=spvae1l3(iv2)
      acd203(27)=abb203(57)
      acd203(28)=spvae1l3(iv1)
      acd203(29)=spvae2k2(iv1)
      acd203(30)=spval3e1(iv2)
      acd203(31)=dotproduct(qshift,spvae1e2)
      acd203(32)=spvae1e2(iv2)
      acd203(33)=dotproduct(qshift,spval3e1)
      acd203(34)=dotproduct(qshift,spval4e1)
      acd203(35)=dotproduct(qshift,spval5e1)
      acd203(36)=abb203(73)
      acd203(37)=spval4e1(iv2)
      acd203(38)=spval5e1(iv2)
      acd203(39)=spvae2k2(iv2)
      acd203(40)=spval3e1(iv1)
      acd203(41)=spvae1e2(iv1)
      acd203(42)=spval4e1(iv1)
      acd203(43)=spval5e1(iv1)
      acd203(44)=dotproduct(qshift,spvae2k2)
      acd203(45)=dotproduct(qshift,spvae2l3)
      acd203(46)=abb203(65)
      acd203(47)=abb203(66)
      acd203(48)=spvae2l3(iv2)
      acd203(49)=abb203(61)
      acd203(50)=spvae2l3(iv1)
      acd203(51)=acd203(1)*acd203(3)
      acd203(52)=acd203(19)*acd203(16)
      acd203(51)=acd203(51)+acd203(52)
      acd203(52)=-acd203(2)*acd203(51)
      acd203(53)=acd203(14)*acd203(3)
      acd203(54)=acd203(19)*acd203(5)
      acd203(53)=acd203(53)+acd203(54)
      acd203(54)=-acd203(15)*acd203(53)
      acd203(55)=acd203(1)*acd203(5)
      acd203(56)=acd203(14)*acd203(16)
      acd203(55)=acd203(55)+acd203(56)
      acd203(56)=-acd203(6)*acd203(55)
      acd203(57)=acd203(29)*acd203(31)
      acd203(58)=acd203(44)*acd203(41)
      acd203(57)=acd203(57)+acd203(58)
      acd203(58)=-acd203(30)*acd203(57)
      acd203(59)=acd203(29)*acd203(32)
      acd203(60)=acd203(39)*acd203(41)
      acd203(59)=acd203(59)+acd203(60)
      acd203(60)=-acd203(33)*acd203(59)
      acd203(61)=acd203(39)*acd203(31)
      acd203(62)=acd203(44)*acd203(32)
      acd203(61)=acd203(61)+acd203(62)
      acd203(62)=-acd203(40)*acd203(61)
      acd203(52)=acd203(62)+acd203(60)+acd203(58)+acd203(56)+acd203(54)+acd203(&
      &52)
      acd203(52)=acd203(4)*acd203(52)
      acd203(54)=-acd203(12)*acd203(51)
      acd203(56)=-acd203(17)*acd203(53)
      acd203(58)=-acd203(37)*acd203(57)
      acd203(60)=-acd203(42)*acd203(61)
      acd203(54)=acd203(60)+acd203(58)+acd203(56)+acd203(54)
      acd203(54)=acd203(8)*acd203(54)
      acd203(51)=-acd203(13)*acd203(51)
      acd203(53)=-acd203(18)*acd203(53)
      acd203(56)=-acd203(38)*acd203(57)
      acd203(57)=-acd203(43)*acd203(61)
      acd203(51)=acd203(57)+acd203(56)+acd203(53)+acd203(51)
      acd203(51)=acd203(10)*acd203(51)
      acd203(53)=-acd203(17)*acd203(22)
      acd203(56)=-acd203(18)*acd203(24)
      acd203(53)=acd203(53)+acd203(56)
      acd203(53)=acd203(26)*acd203(53)
      acd203(56)=-acd203(12)*acd203(22)
      acd203(57)=-acd203(13)*acd203(24)
      acd203(56)=acd203(56)+acd203(57)
      acd203(56)=acd203(28)*acd203(56)
      acd203(53)=acd203(56)+acd203(53)
      acd203(53)=acd203(3)*acd203(53)
      acd203(56)=acd203(42)*acd203(22)
      acd203(57)=acd203(43)*acd203(24)
      acd203(56)=acd203(56)+acd203(57)
      acd203(57)=-acd203(48)*acd203(56)
      acd203(58)=acd203(37)*acd203(22)
      acd203(60)=acd203(38)*acd203(24)
      acd203(58)=acd203(58)+acd203(60)
      acd203(60)=-acd203(50)*acd203(58)
      acd203(57)=acd203(60)+acd203(57)
      acd203(57)=acd203(31)*acd203(57)
      acd203(60)=-acd203(8)*acd203(55)
      acd203(61)=acd203(26)*acd203(16)
      acd203(62)=acd203(28)*acd203(5)
      acd203(61)=acd203(61)+acd203(62)
      acd203(62)=-acd203(22)*acd203(61)
      acd203(60)=acd203(62)+acd203(60)
      acd203(60)=acd203(7)*acd203(60)
      acd203(62)=-acd203(10)*acd203(55)
      acd203(63)=-acd203(24)*acd203(61)
      acd203(62)=acd203(63)+acd203(62)
      acd203(62)=acd203(9)*acd203(62)
      acd203(63)=acd203(12)*acd203(16)
      acd203(64)=acd203(17)*acd203(5)
      acd203(63)=acd203(63)+acd203(64)
      acd203(64)=-acd203(22)*acd203(63)
      acd203(65)=acd203(13)*acd203(16)
      acd203(66)=acd203(18)*acd203(5)
      acd203(65)=acd203(65)+acd203(66)
      acd203(66)=-acd203(24)*acd203(65)
      acd203(64)=acd203(66)+acd203(64)
      acd203(64)=acd203(21)*acd203(64)
      acd203(66)=-acd203(8)*acd203(59)
      acd203(67)=acd203(48)*acd203(41)
      acd203(68)=acd203(50)*acd203(32)
      acd203(67)=acd203(67)+acd203(68)
      acd203(68)=-acd203(22)*acd203(67)
      acd203(66)=acd203(68)+acd203(66)
      acd203(66)=acd203(34)*acd203(66)
      acd203(68)=-acd203(10)*acd203(59)
      acd203(69)=-acd203(24)*acd203(67)
      acd203(68)=acd203(69)+acd203(68)
      acd203(68)=acd203(35)*acd203(68)
      acd203(56)=-acd203(32)*acd203(56)
      acd203(58)=-acd203(41)*acd203(58)
      acd203(56)=acd203(58)+acd203(56)
      acd203(56)=acd203(45)*acd203(56)
      acd203(55)=-acd203(11)*acd203(55)
      acd203(58)=acd203(2)*acd203(16)
      acd203(69)=acd203(15)*acd203(5)
      acd203(58)=acd203(58)+acd203(69)
      acd203(58)=acd203(20)*acd203(58)
      acd203(63)=acd203(23)*acd203(63)
      acd203(65)=acd203(25)*acd203(65)
      acd203(61)=-acd203(27)*acd203(61)
      acd203(59)=acd203(36)*acd203(59)
      acd203(69)=acd203(42)*acd203(32)
      acd203(70)=acd203(37)*acd203(41)
      acd203(69)=acd203(70)+acd203(69)
      acd203(69)=acd203(46)*acd203(69)
      acd203(70)=acd203(43)*acd203(32)
      acd203(71)=acd203(38)*acd203(41)
      acd203(70)=acd203(71)+acd203(70)
      acd203(70)=acd203(47)*acd203(70)
      acd203(67)=acd203(49)*acd203(67)
      brack=acd203(51)+acd203(52)+acd203(53)+acd203(54)+acd203(55)+acd203(56)+a&
      &cd203(57)+acd203(58)+acd203(59)+acd203(60)+acd203(61)+acd203(62)+acd203(&
      &63)+acd203(64)+acd203(65)+acd203(66)+acd203(67)+acd203(68)+acd203(69)+ac&
      &d203(70)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd203h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(54) :: acd203
      complex(ki) :: brack
      acd203(1)=spvae1k2(iv1)
      acd203(2)=spval3e2(iv2)
      acd203(3)=spvae2e1(iv3)
      acd203(4)=abb203(47)
      acd203(5)=spval3e2(iv3)
      acd203(6)=spvae2e1(iv2)
      acd203(7)=spval4e2(iv3)
      acd203(8)=abb203(68)
      acd203(9)=spval5e2(iv3)
      acd203(10)=abb203(72)
      acd203(11)=spval4e2(iv2)
      acd203(12)=spval5e2(iv2)
      acd203(13)=spvae1k2(iv2)
      acd203(14)=spval3e2(iv1)
      acd203(15)=spvae2e1(iv1)
      acd203(16)=spval4e2(iv1)
      acd203(17)=spval5e2(iv1)
      acd203(18)=spvae1k2(iv3)
      acd203(19)=spvae1l3(iv3)
      acd203(20)=abb203(49)
      acd203(21)=spvae1l3(iv2)
      acd203(22)=abb203(52)
      acd203(23)=spvae1l3(iv1)
      acd203(24)=spvae2k2(iv1)
      acd203(25)=spval3e1(iv2)
      acd203(26)=spvae1e2(iv3)
      acd203(27)=spval3e1(iv3)
      acd203(28)=spvae1e2(iv2)
      acd203(29)=spval4e1(iv3)
      acd203(30)=spval5e1(iv3)
      acd203(31)=spval4e1(iv2)
      acd203(32)=spval5e1(iv2)
      acd203(33)=spvae2k2(iv2)
      acd203(34)=spval3e1(iv1)
      acd203(35)=spvae1e2(iv1)
      acd203(36)=spval4e1(iv1)
      acd203(37)=spval5e1(iv1)
      acd203(38)=spvae2k2(iv3)
      acd203(39)=spvae2l3(iv3)
      acd203(40)=spvae2l3(iv2)
      acd203(41)=spvae2l3(iv1)
      acd203(42)=acd203(1)*acd203(3)
      acd203(43)=acd203(18)*acd203(15)
      acd203(42)=acd203(42)+acd203(43)
      acd203(43)=-acd203(2)*acd203(42)
      acd203(44)=acd203(1)*acd203(6)
      acd203(45)=acd203(13)*acd203(15)
      acd203(44)=acd203(44)+acd203(45)
      acd203(45)=-acd203(5)*acd203(44)
      acd203(46)=acd203(13)*acd203(3)
      acd203(47)=acd203(18)*acd203(6)
      acd203(46)=acd203(46)+acd203(47)
      acd203(47)=-acd203(14)*acd203(46)
      acd203(48)=acd203(24)*acd203(26)
      acd203(49)=acd203(38)*acd203(35)
      acd203(48)=acd203(48)+acd203(49)
      acd203(49)=-acd203(25)*acd203(48)
      acd203(50)=acd203(24)*acd203(28)
      acd203(51)=acd203(33)*acd203(35)
      acd203(50)=acd203(50)+acd203(51)
      acd203(51)=-acd203(27)*acd203(50)
      acd203(52)=acd203(33)*acd203(26)
      acd203(53)=acd203(38)*acd203(28)
      acd203(52)=acd203(52)+acd203(53)
      acd203(53)=-acd203(34)*acd203(52)
      acd203(43)=acd203(53)+acd203(51)+acd203(49)+acd203(47)+acd203(45)+acd203(&
      &43)
      acd203(43)=acd203(4)*acd203(43)
      acd203(45)=-acd203(7)*acd203(44)
      acd203(47)=-acd203(11)*acd203(42)
      acd203(49)=-acd203(16)*acd203(46)
      acd203(51)=-acd203(29)*acd203(50)
      acd203(53)=-acd203(31)*acd203(48)
      acd203(54)=-acd203(36)*acd203(52)
      acd203(45)=acd203(54)+acd203(53)+acd203(51)+acd203(49)+acd203(47)+acd203(&
      &45)
      acd203(45)=acd203(8)*acd203(45)
      acd203(44)=-acd203(9)*acd203(44)
      acd203(42)=-acd203(12)*acd203(42)
      acd203(46)=-acd203(17)*acd203(46)
      acd203(47)=-acd203(30)*acd203(50)
      acd203(48)=-acd203(32)*acd203(48)
      acd203(49)=-acd203(37)*acd203(52)
      acd203(42)=acd203(49)+acd203(48)+acd203(47)+acd203(46)+acd203(42)+acd203(&
      &44)
      acd203(42)=acd203(10)*acd203(42)
      acd203(44)=acd203(16)*acd203(20)
      acd203(46)=acd203(17)*acd203(22)
      acd203(44)=acd203(44)+acd203(46)
      acd203(46)=-acd203(6)*acd203(44)
      acd203(47)=acd203(11)*acd203(20)
      acd203(48)=acd203(12)*acd203(22)
      acd203(47)=acd203(47)+acd203(48)
      acd203(48)=-acd203(15)*acd203(47)
      acd203(46)=acd203(48)+acd203(46)
      acd203(46)=acd203(19)*acd203(46)
      acd203(44)=-acd203(3)*acd203(44)
      acd203(48)=acd203(7)*acd203(20)
      acd203(49)=acd203(9)*acd203(22)
      acd203(48)=acd203(48)+acd203(49)
      acd203(49)=-acd203(15)*acd203(48)
      acd203(44)=acd203(49)+acd203(44)
      acd203(44)=acd203(21)*acd203(44)
      acd203(47)=-acd203(3)*acd203(47)
      acd203(48)=-acd203(6)*acd203(48)
      acd203(47)=acd203(48)+acd203(47)
      acd203(47)=acd203(23)*acd203(47)
      acd203(48)=acd203(36)*acd203(20)
      acd203(49)=acd203(37)*acd203(22)
      acd203(48)=acd203(48)+acd203(49)
      acd203(49)=-acd203(28)*acd203(48)
      acd203(50)=acd203(31)*acd203(20)
      acd203(51)=acd203(32)*acd203(22)
      acd203(50)=acd203(50)+acd203(51)
      acd203(51)=-acd203(35)*acd203(50)
      acd203(49)=acd203(51)+acd203(49)
      acd203(49)=acd203(39)*acd203(49)
      acd203(48)=-acd203(26)*acd203(48)
      acd203(51)=acd203(29)*acd203(20)
      acd203(52)=acd203(30)*acd203(22)
      acd203(51)=acd203(51)+acd203(52)
      acd203(52)=-acd203(35)*acd203(51)
      acd203(48)=acd203(52)+acd203(48)
      acd203(48)=acd203(40)*acd203(48)
      acd203(50)=-acd203(26)*acd203(50)
      acd203(51)=-acd203(28)*acd203(51)
      acd203(50)=acd203(51)+acd203(50)
      acd203(50)=acd203(41)*acd203(50)
      brack=acd203(42)+acd203(43)+acd203(44)+acd203(45)+acd203(46)+acd203(47)+a&
      &cd203(48)+acd203(49)+acd203(50)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd203h0
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
      qshift = k2-k4
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
end module     p2_gg_httbar_d203h0l1d
