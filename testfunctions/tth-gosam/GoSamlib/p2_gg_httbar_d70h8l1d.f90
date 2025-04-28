module     p2_gg_httbar_d70h8l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d70h8l1d.f90
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
      use p2_gg_httbar_abbrevd70h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd70
      complex(ki) :: brack
      acd70(1)=dotproduct(qshift,qshift)
      acd70(2)=dotproduct(qshift,spvak1e2)
      acd70(3)=abb70(17)
      acd70(4)=dotproduct(qshift,spvae2k1)
      acd70(5)=abb70(31)
      acd70(6)=dotproduct(qshift,spvak2e2)
      acd70(7)=abb70(22)
      acd70(8)=dotproduct(qshift,spval4e2)
      acd70(9)=abb70(40)
      acd70(10)=dotproduct(qshift,spvae2l4)
      acd70(11)=abb70(35)
      acd70(12)=dotproduct(qshift,spvae2l5)
      acd70(13)=abb70(25)
      acd70(14)=dotproduct(qshift,spvae1e2)
      acd70(15)=abb70(39)
      acd70(16)=dotproduct(qshift,spvae2e1)
      acd70(17)=abb70(12)
      acd70(18)=abb70(20)
      acd70(19)=abb70(36)
      acd70(20)=dotproduct(qshift,spvae2l3)
      acd70(21)=abb70(16)
      acd70(22)=abb70(14)
      acd70(23)=abb70(34)
      acd70(24)=dotproduct(qshift,spval3e2)
      acd70(25)=abb70(32)
      acd70(26)=abb70(26)
      acd70(27)=abb70(10)
      acd70(28)=abb70(23)
      acd70(29)=abb70(9)
      acd70(30)=abb70(15)
      acd70(31)=dotproduct(qshift,spvae2k2)
      acd70(32)=abb70(28)
      acd70(33)=abb70(11)
      acd70(34)=abb70(45)
      acd70(35)=abb70(54)
      acd70(36)=abb70(37)
      acd70(37)=abb70(48)
      acd70(38)=abb70(30)
      acd70(39)=abb70(18)
      acd70(40)=abb70(24)
      acd70(41)=abb70(21)
      acd70(42)=abb70(51)
      acd70(43)=abb70(29)
      acd70(44)=abb70(33)
      acd70(45)=abb70(27)
      acd70(46)=abb70(49)
      acd70(47)=abb70(19)
      acd70(48)=abb70(52)
      acd70(49)=abb70(13)
      acd70(50)=-acd70(3)*acd70(2)
      acd70(51)=-acd70(5)*acd70(4)
      acd70(52)=-acd70(7)*acd70(6)
      acd70(53)=acd70(9)*acd70(8)
      acd70(54)=-acd70(11)*acd70(10)
      acd70(55)=-acd70(13)*acd70(12)
      acd70(56)=-acd70(15)*acd70(14)
      acd70(57)=acd70(17)*acd70(16)
      acd70(50)=acd70(18)+acd70(57)+acd70(56)+acd70(55)+acd70(54)+acd70(53)+acd&
      &70(52)+acd70(50)+acd70(51)
      acd70(50)=acd70(1)*acd70(50)
      acd70(51)=acd70(23)*acd70(4)
      acd70(52)=acd70(27)*acd70(10)
      acd70(53)=acd70(28)*acd70(12)
      acd70(54)=acd70(29)*acd70(16)
      acd70(55)=acd70(30)*acd70(20)
      acd70(56)=acd70(32)*acd70(31)
      acd70(51)=-acd70(33)+acd70(56)+acd70(55)+acd70(54)+acd70(53)+acd70(52)+ac&
      &d70(51)
      acd70(51)=acd70(6)*acd70(51)
      acd70(52)=acd70(25)*acd70(4)
      acd70(53)=acd70(37)*acd70(10)
      acd70(54)=acd70(40)*acd70(12)
      acd70(55)=acd70(44)*acd70(16)
      acd70(52)=-acd70(47)+acd70(55)+acd70(54)+acd70(53)+acd70(52)
      acd70(52)=acd70(24)*acd70(52)
      acd70(53)=acd70(19)*acd70(2)
      acd70(54)=acd70(34)*acd70(8)
      acd70(55)=acd70(39)*acd70(14)
      acd70(53)=-acd70(41)+acd70(55)+acd70(54)+acd70(53)
      acd70(53)=acd70(12)*acd70(53)
      acd70(54)=acd70(21)*acd70(2)
      acd70(55)=acd70(35)*acd70(8)
      acd70(56)=acd70(42)*acd70(14)
      acd70(54)=-acd70(46)+acd70(56)+acd70(55)+acd70(54)
      acd70(54)=acd70(20)*acd70(54)
      acd70(55)=-acd70(22)*acd70(2)
      acd70(56)=-acd70(26)*acd70(4)
      acd70(57)=-acd70(36)*acd70(8)
      acd70(58)=-acd70(38)*acd70(10)
      acd70(59)=-acd70(43)*acd70(14)
      acd70(60)=-acd70(45)*acd70(16)
      acd70(61)=-acd70(48)*acd70(31)
      brack=acd70(49)+acd70(50)+acd70(51)+acd70(52)+acd70(53)+acd70(54)+acd70(5&
      &5)+acd70(56)+acd70(57)+acd70(58)+acd70(59)+acd70(60)+acd70(61)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd70h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(77) :: acd70
      complex(ki) :: brack
      acd70(1)=qshift(iv1)
      acd70(2)=dotproduct(qshift,spvak1e2)
      acd70(3)=abb70(17)
      acd70(4)=dotproduct(qshift,spvae2k1)
      acd70(5)=abb70(31)
      acd70(6)=dotproduct(qshift,spvak2e2)
      acd70(7)=abb70(22)
      acd70(8)=dotproduct(qshift,spval4e2)
      acd70(9)=abb70(40)
      acd70(10)=dotproduct(qshift,spvae2l4)
      acd70(11)=abb70(35)
      acd70(12)=dotproduct(qshift,spvae2l5)
      acd70(13)=abb70(25)
      acd70(14)=dotproduct(qshift,spvae1e2)
      acd70(15)=abb70(39)
      acd70(16)=dotproduct(qshift,spvae2e1)
      acd70(17)=abb70(12)
      acd70(18)=abb70(20)
      acd70(19)=spvak1e2(iv1)
      acd70(20)=dotproduct(qshift,qshift)
      acd70(21)=abb70(36)
      acd70(22)=dotproduct(qshift,spvae2l3)
      acd70(23)=abb70(16)
      acd70(24)=abb70(14)
      acd70(25)=spvae2k1(iv1)
      acd70(26)=abb70(34)
      acd70(27)=dotproduct(qshift,spval3e2)
      acd70(28)=abb70(32)
      acd70(29)=abb70(26)
      acd70(30)=spvak2e2(iv1)
      acd70(31)=abb70(10)
      acd70(32)=abb70(23)
      acd70(33)=abb70(9)
      acd70(34)=abb70(15)
      acd70(35)=dotproduct(qshift,spvae2k2)
      acd70(36)=abb70(28)
      acd70(37)=abb70(11)
      acd70(38)=spval4e2(iv1)
      acd70(39)=abb70(45)
      acd70(40)=abb70(54)
      acd70(41)=abb70(37)
      acd70(42)=spvae2l4(iv1)
      acd70(43)=abb70(48)
      acd70(44)=abb70(30)
      acd70(45)=spvae2l5(iv1)
      acd70(46)=abb70(18)
      acd70(47)=abb70(24)
      acd70(48)=abb70(21)
      acd70(49)=spvae1e2(iv1)
      acd70(50)=abb70(51)
      acd70(51)=abb70(29)
      acd70(52)=spvae2e1(iv1)
      acd70(53)=abb70(33)
      acd70(54)=abb70(27)
      acd70(55)=spvae2l3(iv1)
      acd70(56)=abb70(49)
      acd70(57)=spval3e2(iv1)
      acd70(58)=abb70(19)
      acd70(59)=spvae2k2(iv1)
      acd70(60)=abb70(52)
      acd70(61)=acd70(52)*acd70(17)
      acd70(62)=-acd70(49)*acd70(15)
      acd70(63)=-acd70(42)*acd70(11)
      acd70(64)=acd70(38)*acd70(9)
      acd70(65)=-acd70(25)*acd70(5)
      acd70(66)=-acd70(19)*acd70(3)
      acd70(67)=-acd70(45)*acd70(13)
      acd70(68)=-acd70(30)*acd70(7)
      acd70(61)=acd70(68)+acd70(67)+acd70(66)+acd70(65)+acd70(64)+acd70(63)+acd&
      &70(61)+acd70(62)
      acd70(61)=acd70(20)*acd70(61)
      acd70(62)=acd70(16)*acd70(17)
      acd70(63)=-acd70(14)*acd70(15)
      acd70(64)=-acd70(10)*acd70(11)
      acd70(65)=acd70(8)*acd70(9)
      acd70(66)=-acd70(4)*acd70(5)
      acd70(67)=-acd70(2)*acd70(3)
      acd70(68)=-acd70(12)*acd70(13)
      acd70(69)=-acd70(6)*acd70(7)
      acd70(62)=acd70(69)+acd70(68)+acd70(67)+acd70(66)+acd70(65)+acd70(64)+acd&
      &70(63)+acd70(18)+acd70(62)
      acd70(62)=acd70(1)*acd70(62)
      acd70(63)=acd70(36)*acd70(59)
      acd70(64)=acd70(52)*acd70(33)
      acd70(65)=acd70(42)*acd70(31)
      acd70(66)=acd70(25)*acd70(26)
      acd70(67)=acd70(55)*acd70(34)
      acd70(68)=acd70(45)*acd70(32)
      acd70(63)=acd70(68)+acd70(67)+acd70(66)+acd70(65)+acd70(63)+acd70(64)
      acd70(63)=acd70(6)*acd70(63)
      acd70(64)=acd70(36)*acd70(35)
      acd70(65)=acd70(16)*acd70(33)
      acd70(66)=acd70(10)*acd70(31)
      acd70(67)=acd70(4)*acd70(26)
      acd70(68)=acd70(22)*acd70(34)
      acd70(69)=acd70(12)*acd70(32)
      acd70(64)=acd70(69)+acd70(68)+acd70(67)+acd70(66)+acd70(65)-acd70(37)+acd&
      &70(64)
      acd70(64)=acd70(30)*acd70(64)
      acd70(65)=acd70(49)*acd70(46)
      acd70(66)=acd70(38)*acd70(39)
      acd70(67)=acd70(19)*acd70(21)
      acd70(68)=acd70(57)*acd70(47)
      acd70(65)=acd70(68)+acd70(67)+acd70(65)+acd70(66)
      acd70(65)=acd70(12)*acd70(65)
      acd70(66)=acd70(14)*acd70(46)
      acd70(67)=acd70(8)*acd70(39)
      acd70(68)=acd70(2)*acd70(21)
      acd70(69)=acd70(27)*acd70(47)
      acd70(66)=acd70(69)+acd70(68)+acd70(67)-acd70(48)+acd70(66)
      acd70(66)=acd70(45)*acd70(66)
      acd70(67)=acd70(16)*acd70(53)
      acd70(68)=acd70(10)*acd70(43)
      acd70(69)=acd70(4)*acd70(28)
      acd70(67)=acd70(69)+acd70(68)-acd70(58)+acd70(67)
      acd70(67)=acd70(57)*acd70(67)
      acd70(68)=acd70(14)*acd70(50)
      acd70(69)=acd70(8)*acd70(40)
      acd70(70)=acd70(2)*acd70(23)
      acd70(68)=acd70(70)+acd70(69)-acd70(56)+acd70(68)
      acd70(68)=acd70(55)*acd70(68)
      acd70(69)=acd70(52)*acd70(53)
      acd70(70)=acd70(42)*acd70(43)
      acd70(69)=acd70(69)+acd70(70)
      acd70(69)=acd70(27)*acd70(69)
      acd70(70)=acd70(49)*acd70(50)
      acd70(71)=acd70(38)*acd70(40)
      acd70(70)=acd70(70)+acd70(71)
      acd70(70)=acd70(22)*acd70(70)
      acd70(71)=-acd70(59)*acd70(60)
      acd70(72)=-acd70(52)*acd70(54)
      acd70(73)=-acd70(49)*acd70(51)
      acd70(74)=-acd70(42)*acd70(44)
      acd70(75)=-acd70(38)*acd70(41)
      acd70(76)=acd70(27)*acd70(28)
      acd70(76)=-acd70(29)+acd70(76)
      acd70(76)=acd70(25)*acd70(76)
      acd70(77)=acd70(22)*acd70(23)
      acd70(77)=-acd70(24)+acd70(77)
      acd70(77)=acd70(19)*acd70(77)
      brack=acd70(61)+2.0_ki*acd70(62)+acd70(63)+acd70(64)+acd70(65)+acd70(66)+&
      &acd70(67)+acd70(68)+acd70(69)+acd70(70)+acd70(71)+acd70(72)+acd70(73)+ac&
      &d70(74)+acd70(75)+acd70(76)+acd70(77)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd70h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd70
      complex(ki) :: brack
      acd70(1)=d(iv1,iv2)
      acd70(2)=dotproduct(qshift,spvak1e2)
      acd70(3)=abb70(17)
      acd70(4)=dotproduct(qshift,spvae2k1)
      acd70(5)=abb70(31)
      acd70(6)=dotproduct(qshift,spvak2e2)
      acd70(7)=abb70(22)
      acd70(8)=dotproduct(qshift,spval4e2)
      acd70(9)=abb70(40)
      acd70(10)=dotproduct(qshift,spvae2l4)
      acd70(11)=abb70(35)
      acd70(12)=dotproduct(qshift,spvae2l5)
      acd70(13)=abb70(25)
      acd70(14)=dotproduct(qshift,spvae1e2)
      acd70(15)=abb70(39)
      acd70(16)=dotproduct(qshift,spvae2e1)
      acd70(17)=abb70(12)
      acd70(18)=abb70(20)
      acd70(19)=qshift(iv1)
      acd70(20)=spvak1e2(iv2)
      acd70(21)=spvae2k1(iv2)
      acd70(22)=spvak2e2(iv2)
      acd70(23)=spval4e2(iv2)
      acd70(24)=spvae2l4(iv2)
      acd70(25)=spvae2l5(iv2)
      acd70(26)=spvae1e2(iv2)
      acd70(27)=spvae2e1(iv2)
      acd70(28)=qshift(iv2)
      acd70(29)=spvak1e2(iv1)
      acd70(30)=spvae2k1(iv1)
      acd70(31)=spvak2e2(iv1)
      acd70(32)=spval4e2(iv1)
      acd70(33)=spvae2l4(iv1)
      acd70(34)=spvae2l5(iv1)
      acd70(35)=spvae1e2(iv1)
      acd70(36)=spvae2e1(iv1)
      acd70(37)=abb70(36)
      acd70(38)=spvae2l3(iv2)
      acd70(39)=abb70(16)
      acd70(40)=spvae2l3(iv1)
      acd70(41)=abb70(34)
      acd70(42)=spval3e2(iv2)
      acd70(43)=abb70(32)
      acd70(44)=spval3e2(iv1)
      acd70(45)=abb70(10)
      acd70(46)=abb70(23)
      acd70(47)=abb70(9)
      acd70(48)=abb70(15)
      acd70(49)=spvae2k2(iv2)
      acd70(50)=abb70(28)
      acd70(51)=spvae2k2(iv1)
      acd70(52)=abb70(45)
      acd70(53)=abb70(54)
      acd70(54)=abb70(48)
      acd70(55)=abb70(18)
      acd70(56)=abb70(24)
      acd70(57)=abb70(51)
      acd70(58)=abb70(33)
      acd70(59)=acd70(17)*acd70(36)
      acd70(60)=-acd70(15)*acd70(35)
      acd70(61)=-acd70(11)*acd70(33)
      acd70(62)=acd70(9)*acd70(32)
      acd70(63)=-acd70(5)*acd70(30)
      acd70(64)=-acd70(3)*acd70(29)
      acd70(65)=-acd70(34)*acd70(13)
      acd70(66)=-acd70(31)*acd70(7)
      acd70(59)=acd70(66)+acd70(65)+acd70(64)+acd70(63)+acd70(62)+acd70(61)+acd&
      &70(59)+acd70(60)
      acd70(59)=acd70(28)*acd70(59)
      acd70(60)=acd70(17)*acd70(27)
      acd70(61)=-acd70(15)*acd70(26)
      acd70(62)=-acd70(11)*acd70(24)
      acd70(63)=acd70(9)*acd70(23)
      acd70(64)=-acd70(5)*acd70(21)
      acd70(65)=-acd70(3)*acd70(20)
      acd70(66)=-acd70(25)*acd70(13)
      acd70(67)=-acd70(22)*acd70(7)
      acd70(60)=acd70(67)+acd70(66)+acd70(65)+acd70(64)+acd70(63)+acd70(62)+acd&
      &70(60)+acd70(61)
      acd70(60)=acd70(19)*acd70(60)
      acd70(61)=acd70(17)*acd70(16)
      acd70(62)=-acd70(15)*acd70(14)
      acd70(63)=-acd70(13)*acd70(12)
      acd70(64)=-acd70(11)*acd70(10)
      acd70(65)=acd70(9)*acd70(8)
      acd70(66)=-acd70(7)*acd70(6)
      acd70(67)=-acd70(5)*acd70(4)
      acd70(68)=-acd70(3)*acd70(2)
      acd70(61)=acd70(68)+acd70(67)+acd70(66)+acd70(65)+acd70(64)+acd70(63)+acd&
      &70(62)+acd70(18)+acd70(61)
      acd70(61)=acd70(1)*acd70(61)
      acd70(59)=acd70(61)+acd70(59)+acd70(60)
      acd70(60)=acd70(50)*acd70(49)
      acd70(61)=acd70(27)*acd70(47)
      acd70(62)=acd70(24)*acd70(45)
      acd70(63)=acd70(21)*acd70(41)
      acd70(64)=acd70(38)*acd70(48)
      acd70(65)=acd70(25)*acd70(46)
      acd70(60)=acd70(65)+acd70(64)+acd70(63)+acd70(62)+acd70(60)+acd70(61)
      acd70(60)=acd70(31)*acd70(60)
      acd70(61)=acd70(50)*acd70(51)
      acd70(62)=acd70(36)*acd70(47)
      acd70(63)=acd70(33)*acd70(45)
      acd70(64)=acd70(30)*acd70(41)
      acd70(65)=acd70(40)*acd70(48)
      acd70(66)=acd70(34)*acd70(46)
      acd70(61)=acd70(66)+acd70(65)+acd70(64)+acd70(63)+acd70(61)+acd70(62)
      acd70(61)=acd70(22)*acd70(61)
      acd70(62)=acd70(26)*acd70(55)
      acd70(63)=acd70(23)*acd70(52)
      acd70(64)=acd70(20)*acd70(37)
      acd70(65)=acd70(42)*acd70(56)
      acd70(62)=acd70(65)+acd70(64)+acd70(62)+acd70(63)
      acd70(62)=acd70(34)*acd70(62)
      acd70(63)=acd70(35)*acd70(55)
      acd70(64)=acd70(32)*acd70(52)
      acd70(65)=acd70(29)*acd70(37)
      acd70(66)=acd70(44)*acd70(56)
      acd70(63)=acd70(66)+acd70(65)+acd70(63)+acd70(64)
      acd70(63)=acd70(25)*acd70(63)
      acd70(64)=acd70(27)*acd70(58)
      acd70(65)=acd70(24)*acd70(54)
      acd70(66)=acd70(21)*acd70(43)
      acd70(64)=acd70(66)+acd70(64)+acd70(65)
      acd70(64)=acd70(44)*acd70(64)
      acd70(65)=acd70(36)*acd70(58)
      acd70(66)=acd70(33)*acd70(54)
      acd70(67)=acd70(30)*acd70(43)
      acd70(65)=acd70(67)+acd70(65)+acd70(66)
      acd70(65)=acd70(42)*acd70(65)
      acd70(66)=acd70(26)*acd70(57)
      acd70(67)=acd70(23)*acd70(53)
      acd70(68)=acd70(20)*acd70(39)
      acd70(66)=acd70(68)+acd70(66)+acd70(67)
      acd70(66)=acd70(40)*acd70(66)
      acd70(67)=acd70(35)*acd70(57)
      acd70(68)=acd70(32)*acd70(53)
      acd70(69)=acd70(29)*acd70(39)
      acd70(67)=acd70(69)+acd70(67)+acd70(68)
      acd70(67)=acd70(38)*acd70(67)
      brack=2.0_ki*acd70(59)+acd70(60)+acd70(61)+acd70(62)+acd70(63)+acd70(64)+&
      &acd70(65)+acd70(66)+acd70(67)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd70h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd70
      complex(ki) :: brack
      acd70(1)=d(iv1,iv2)
      acd70(2)=spvak1e2(iv3)
      acd70(3)=abb70(17)
      acd70(4)=spvae2k1(iv3)
      acd70(5)=abb70(31)
      acd70(6)=spvak2e2(iv3)
      acd70(7)=abb70(22)
      acd70(8)=spval4e2(iv3)
      acd70(9)=abb70(40)
      acd70(10)=spvae2l4(iv3)
      acd70(11)=abb70(35)
      acd70(12)=spvae2l5(iv3)
      acd70(13)=abb70(25)
      acd70(14)=spvae1e2(iv3)
      acd70(15)=abb70(39)
      acd70(16)=spvae2e1(iv3)
      acd70(17)=abb70(12)
      acd70(18)=d(iv1,iv3)
      acd70(19)=spvak1e2(iv2)
      acd70(20)=spvae2k1(iv2)
      acd70(21)=spvak2e2(iv2)
      acd70(22)=spval4e2(iv2)
      acd70(23)=spvae2l4(iv2)
      acd70(24)=spvae2l5(iv2)
      acd70(25)=spvae1e2(iv2)
      acd70(26)=spvae2e1(iv2)
      acd70(27)=d(iv2,iv3)
      acd70(28)=spvak1e2(iv1)
      acd70(29)=spvae2k1(iv1)
      acd70(30)=spvak2e2(iv1)
      acd70(31)=spval4e2(iv1)
      acd70(32)=spvae2l4(iv1)
      acd70(33)=spvae2l5(iv1)
      acd70(34)=spvae1e2(iv1)
      acd70(35)=spvae2e1(iv1)
      acd70(36)=-acd70(2)*acd70(3)
      acd70(37)=-acd70(4)*acd70(5)
      acd70(38)=-acd70(6)*acd70(7)
      acd70(39)=acd70(8)*acd70(9)
      acd70(40)=-acd70(10)*acd70(11)
      acd70(41)=-acd70(12)*acd70(13)
      acd70(42)=-acd70(14)*acd70(15)
      acd70(43)=acd70(16)*acd70(17)
      acd70(36)=acd70(43)+acd70(42)+acd70(41)+acd70(40)+acd70(39)+acd70(38)+acd&
      &70(36)+acd70(37)
      acd70(36)=acd70(1)*acd70(36)
      acd70(37)=-acd70(19)*acd70(3)
      acd70(38)=-acd70(20)*acd70(5)
      acd70(39)=-acd70(21)*acd70(7)
      acd70(40)=acd70(22)*acd70(9)
      acd70(41)=-acd70(23)*acd70(11)
      acd70(42)=-acd70(24)*acd70(13)
      acd70(43)=-acd70(25)*acd70(15)
      acd70(44)=acd70(26)*acd70(17)
      acd70(37)=acd70(44)+acd70(43)+acd70(42)+acd70(41)+acd70(40)+acd70(39)+acd&
      &70(38)+acd70(37)
      acd70(37)=acd70(18)*acd70(37)
      acd70(38)=-acd70(28)*acd70(3)
      acd70(39)=-acd70(29)*acd70(5)
      acd70(40)=-acd70(30)*acd70(7)
      acd70(41)=acd70(31)*acd70(9)
      acd70(42)=-acd70(32)*acd70(11)
      acd70(43)=-acd70(33)*acd70(13)
      acd70(44)=-acd70(34)*acd70(15)
      acd70(45)=acd70(35)*acd70(17)
      acd70(38)=acd70(45)+acd70(44)+acd70(43)+acd70(42)+acd70(41)+acd70(40)+acd&
      &70(39)+acd70(38)
      acd70(38)=acd70(27)*acd70(38)
      acd70(36)=acd70(38)+acd70(37)+acd70(36)
      brack=2.0_ki*acd70(36)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd70h8
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
end module     p2_gg_httbar_d70h8l1d
