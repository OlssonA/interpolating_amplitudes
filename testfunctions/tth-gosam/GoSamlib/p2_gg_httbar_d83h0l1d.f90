module     p2_gg_httbar_d83h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d83h0l1d.f90
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
      use p2_gg_httbar_abbrevd83h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(68) :: acd83
      complex(ki) :: brack
      acd83(1)=dotproduct(qshift,qshift)
      acd83(2)=abb83(22)
      acd83(3)=dotproduct(qshift,spvak1k2)
      acd83(4)=abb83(25)
      acd83(5)=dotproduct(qshift,spvak1l3)
      acd83(6)=abb83(48)
      acd83(7)=dotproduct(qshift,spval3k1)
      acd83(8)=abb83(15)
      acd83(9)=dotproduct(qshift,spval4k1)
      acd83(10)=abb83(18)
      acd83(11)=dotproduct(qshift,spval5k1)
      acd83(12)=abb83(16)
      acd83(13)=dotproduct(qshift,spvak2e1)
      acd83(14)=abb83(23)
      acd83(15)=dotproduct(qshift,spvae1k2)
      acd83(16)=dotproduct(qshift,spval3e2)
      acd83(17)=dotproduct(qshift,spvae2e1)
      acd83(18)=abb83(28)
      acd83(19)=dotproduct(qshift,spval4e2)
      acd83(20)=abb83(32)
      acd83(21)=abb83(19)
      acd83(22)=abb83(14)
      acd83(23)=abb83(20)
      acd83(24)=abb83(8)
      acd83(25)=abb83(35)
      acd83(26)=dotproduct(qshift,spvae1l3)
      acd83(27)=abb83(39)
      acd83(28)=abb83(26)
      acd83(29)=abb83(24)
      acd83(30)=dotproduct(qshift,spvae2k2)
      acd83(31)=dotproduct(qshift,spval5e1)
      acd83(32)=dotproduct(qshift,spvae1e2)
      acd83(33)=abb83(9)
      acd83(34)=abb83(17)
      acd83(35)=abb83(11)
      acd83(36)=dotproduct(qshift,spvae2l3)
      acd83(37)=abb83(31)
      acd83(38)=abb83(29)
      acd83(39)=abb83(34)
      acd83(40)=dotproduct(qshift,spval3e1)
      acd83(41)=abb83(12)
      acd83(42)=abb83(38)
      acd83(43)=dotproduct(qshift,spval4e1)
      acd83(44)=abb83(10)
      acd83(45)=abb83(13)
      acd83(46)=abb83(33)
      acd83(47)=abb83(30)
      acd83(48)=abb83(43)
      acd83(49)=abb83(36)
      acd83(50)=abb83(21)
      acd83(51)=acd83(33)*acd83(30)
      acd83(52)=-acd83(37)*acd83(36)
      acd83(51)=acd83(38)+acd83(52)+acd83(51)
      acd83(51)=acd83(51)*acd83(31)
      acd83(52)=acd83(34)*acd83(30)
      acd83(53)=acd83(41)*acd83(40)
      acd83(54)=acd83(42)*acd83(36)
      acd83(55)=acd83(44)*acd83(43)
      acd83(51)=acd83(54)+acd83(52)+acd83(51)-acd83(45)+acd83(55)+acd83(53)
      acd83(51)=acd83(32)*acd83(51)
      acd83(52)=acd83(18)*acd83(16)
      acd83(53)=acd83(20)*acd83(19)
      acd83(52)=acd83(21)+acd83(53)+acd83(52)
      acd83(52)=acd83(52)*acd83(15)
      acd83(53)=acd83(23)*acd83(16)
      acd83(54)=acd83(25)*acd83(19)
      acd83(52)=-acd83(28)+acd83(54)+acd83(53)+acd83(52)
      acd83(52)=acd83(17)*acd83(52)
      acd83(53)=acd83(27)*acd83(17)
      acd83(53)=-acd83(47)+acd83(53)
      acd83(53)=acd83(26)*acd83(53)
      acd83(54)=acd83(2)*acd83(1)
      acd83(55)=-acd83(4)*acd83(3)
      acd83(56)=acd83(6)*acd83(5)
      acd83(57)=-acd83(8)*acd83(7)
      acd83(58)=-acd83(10)*acd83(9)
      acd83(59)=-acd83(12)*acd83(11)
      acd83(60)=-acd83(14)*acd83(13)
      acd83(61)=-acd83(22)*acd83(15)
      acd83(62)=-acd83(24)*acd83(16)
      acd83(63)=-acd83(29)*acd83(19)
      acd83(64)=-acd83(35)*acd83(30)
      acd83(65)=-acd83(39)*acd83(31)
      acd83(66)=-acd83(46)*acd83(40)
      acd83(67)=-acd83(48)*acd83(36)
      acd83(68)=-acd83(49)*acd83(43)
      brack=acd83(50)+acd83(51)+acd83(52)+acd83(53)+acd83(54)+acd83(55)+acd83(5&
      &6)+acd83(57)+acd83(58)+acd83(59)+acd83(60)+acd83(61)+acd83(62)+acd83(63)&
      &+acd83(64)+acd83(65)+acd83(66)+acd83(67)+acd83(68)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd83h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(80) :: acd83
      complex(ki) :: brack
      acd83(1)=qshift(iv1)
      acd83(2)=abb83(22)
      acd83(3)=spvak1k2(iv1)
      acd83(4)=abb83(25)
      acd83(5)=spvak1l3(iv1)
      acd83(6)=abb83(48)
      acd83(7)=spval3k1(iv1)
      acd83(8)=abb83(15)
      acd83(9)=spval4k1(iv1)
      acd83(10)=abb83(18)
      acd83(11)=spval5k1(iv1)
      acd83(12)=abb83(16)
      acd83(13)=spvak2e1(iv1)
      acd83(14)=abb83(23)
      acd83(15)=spvae1k2(iv1)
      acd83(16)=dotproduct(qshift,spval3e2)
      acd83(17)=dotproduct(qshift,spvae2e1)
      acd83(18)=abb83(28)
      acd83(19)=dotproduct(qshift,spval4e2)
      acd83(20)=abb83(32)
      acd83(21)=abb83(19)
      acd83(22)=abb83(14)
      acd83(23)=spval3e2(iv1)
      acd83(24)=dotproduct(qshift,spvae1k2)
      acd83(25)=abb83(20)
      acd83(26)=abb83(8)
      acd83(27)=spvae2e1(iv1)
      acd83(28)=abb83(35)
      acd83(29)=dotproduct(qshift,spvae1l3)
      acd83(30)=abb83(39)
      acd83(31)=abb83(26)
      acd83(32)=spval4e2(iv1)
      acd83(33)=abb83(24)
      acd83(34)=spvae2k2(iv1)
      acd83(35)=dotproduct(qshift,spval5e1)
      acd83(36)=dotproduct(qshift,spvae1e2)
      acd83(37)=abb83(9)
      acd83(38)=abb83(17)
      acd83(39)=abb83(11)
      acd83(40)=spval5e1(iv1)
      acd83(41)=dotproduct(qshift,spvae2k2)
      acd83(42)=dotproduct(qshift,spvae2l3)
      acd83(43)=abb83(31)
      acd83(44)=abb83(29)
      acd83(45)=abb83(34)
      acd83(46)=spvae1e2(iv1)
      acd83(47)=dotproduct(qshift,spval3e1)
      acd83(48)=abb83(12)
      acd83(49)=abb83(38)
      acd83(50)=dotproduct(qshift,spval4e1)
      acd83(51)=abb83(10)
      acd83(52)=abb83(13)
      acd83(53)=spval3e1(iv1)
      acd83(54)=abb83(33)
      acd83(55)=spvae1l3(iv1)
      acd83(56)=abb83(30)
      acd83(57)=spvae2l3(iv1)
      acd83(58)=abb83(43)
      acd83(59)=spval4e1(iv1)
      acd83(60)=abb83(36)
      acd83(61)=acd83(42)*acd83(43)
      acd83(62)=acd83(37)*acd83(41)
      acd83(61)=-acd83(44)+acd83(61)-acd83(62)
      acd83(62)=-acd83(40)*acd83(61)
      acd83(63)=-acd83(43)*acd83(57)
      acd83(64)=acd83(34)*acd83(37)
      acd83(63)=acd83(63)+acd83(64)
      acd83(63)=acd83(35)*acd83(63)
      acd83(64)=acd83(51)*acd83(59)
      acd83(65)=acd83(48)*acd83(53)
      acd83(66)=acd83(57)*acd83(49)
      acd83(67)=acd83(34)*acd83(38)
      acd83(62)=acd83(63)+acd83(62)+acd83(67)+acd83(66)+acd83(64)+acd83(65)
      acd83(62)=acd83(36)*acd83(62)
      acd83(61)=-acd83(35)*acd83(61)
      acd83(63)=acd83(51)*acd83(50)
      acd83(64)=acd83(48)*acd83(47)
      acd83(65)=acd83(42)*acd83(49)
      acd83(66)=acd83(41)*acd83(38)
      acd83(61)=acd83(61)+acd83(66)+acd83(65)+acd83(64)-acd83(52)+acd83(63)
      acd83(61)=acd83(46)*acd83(61)
      acd83(63)=acd83(19)*acd83(20)
      acd83(64)=acd83(16)*acd83(18)
      acd83(63)=acd83(21)+acd83(63)+acd83(64)
      acd83(64)=acd83(15)*acd83(63)
      acd83(65)=acd83(20)*acd83(32)
      acd83(66)=acd83(18)*acd83(23)
      acd83(65)=acd83(65)+acd83(66)
      acd83(65)=acd83(24)*acd83(65)
      acd83(66)=acd83(30)*acd83(55)
      acd83(67)=acd83(32)*acd83(28)
      acd83(68)=acd83(23)*acd83(25)
      acd83(64)=acd83(65)+acd83(64)+acd83(68)+acd83(66)+acd83(67)
      acd83(64)=acd83(17)*acd83(64)
      acd83(63)=acd83(24)*acd83(63)
      acd83(65)=acd83(30)*acd83(29)
      acd83(66)=acd83(19)*acd83(28)
      acd83(67)=acd83(16)*acd83(25)
      acd83(63)=acd83(63)+acd83(67)+acd83(66)-acd83(31)+acd83(65)
      acd83(63)=acd83(27)*acd83(63)
      acd83(65)=-acd83(13)*acd83(14)
      acd83(66)=-acd83(11)*acd83(12)
      acd83(67)=-acd83(9)*acd83(10)
      acd83(68)=-acd83(7)*acd83(8)
      acd83(69)=acd83(5)*acd83(6)
      acd83(70)=-acd83(3)*acd83(4)
      acd83(71)=acd83(1)*acd83(2)
      acd83(72)=-acd83(59)*acd83(60)
      acd83(73)=-acd83(55)*acd83(56)
      acd83(74)=-acd83(53)*acd83(54)
      acd83(75)=-acd83(57)*acd83(58)
      acd83(76)=-acd83(34)*acd83(39)
      acd83(77)=-acd83(32)*acd83(33)
      acd83(78)=-acd83(23)*acd83(26)
      acd83(79)=-acd83(40)*acd83(45)
      acd83(80)=-acd83(15)*acd83(22)
      brack=acd83(61)+acd83(62)+acd83(63)+acd83(64)+acd83(65)+acd83(66)+acd83(6&
      &7)+acd83(68)+acd83(69)+acd83(70)+2.0_ki*acd83(71)+acd83(72)+acd83(73)+ac&
      &d83(74)+acd83(75)+acd83(76)+acd83(77)+acd83(78)+acd83(79)+acd83(80)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd83h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(61) :: acd83
      complex(ki) :: brack
      acd83(1)=d(iv1,iv2)
      acd83(2)=abb83(22)
      acd83(3)=spvae1k2(iv1)
      acd83(4)=spval3e2(iv2)
      acd83(5)=dotproduct(qshift,spvae2e1)
      acd83(6)=abb83(28)
      acd83(7)=spvae2e1(iv2)
      acd83(8)=dotproduct(qshift,spval3e2)
      acd83(9)=dotproduct(qshift,spval4e2)
      acd83(10)=abb83(32)
      acd83(11)=abb83(19)
      acd83(12)=spval4e2(iv2)
      acd83(13)=spvae1k2(iv2)
      acd83(14)=spval3e2(iv1)
      acd83(15)=spvae2e1(iv1)
      acd83(16)=spval4e2(iv1)
      acd83(17)=dotproduct(qshift,spvae1k2)
      acd83(18)=abb83(20)
      acd83(19)=abb83(35)
      acd83(20)=spvae1l3(iv2)
      acd83(21)=abb83(39)
      acd83(22)=spvae1l3(iv1)
      acd83(23)=spvae2k2(iv1)
      acd83(24)=spval5e1(iv2)
      acd83(25)=dotproduct(qshift,spvae1e2)
      acd83(26)=abb83(9)
      acd83(27)=spvae1e2(iv2)
      acd83(28)=dotproduct(qshift,spval5e1)
      acd83(29)=abb83(17)
      acd83(30)=spvae2k2(iv2)
      acd83(31)=spval5e1(iv1)
      acd83(32)=spvae1e2(iv1)
      acd83(33)=dotproduct(qshift,spvae2k2)
      acd83(34)=dotproduct(qshift,spvae2l3)
      acd83(35)=abb83(31)
      acd83(36)=abb83(29)
      acd83(37)=spvae2l3(iv2)
      acd83(38)=spvae2l3(iv1)
      acd83(39)=spval3e1(iv2)
      acd83(40)=abb83(12)
      acd83(41)=abb83(38)
      acd83(42)=spval4e1(iv2)
      acd83(43)=abb83(10)
      acd83(44)=spval3e1(iv1)
      acd83(45)=spval4e1(iv1)
      acd83(46)=acd83(4)*acd83(6)
      acd83(47)=acd83(12)*acd83(10)
      acd83(46)=acd83(46)+acd83(47)
      acd83(47)=acd83(5)*acd83(3)
      acd83(48)=acd83(17)*acd83(15)
      acd83(47)=acd83(47)+acd83(48)
      acd83(46)=acd83(47)*acd83(46)
      acd83(47)=acd83(14)*acd83(6)
      acd83(48)=acd83(16)*acd83(10)
      acd83(47)=acd83(48)+acd83(47)
      acd83(48)=acd83(5)*acd83(13)
      acd83(49)=acd83(17)*acd83(7)
      acd83(48)=acd83(48)+acd83(49)
      acd83(47)=acd83(48)*acd83(47)
      acd83(48)=acd83(8)*acd83(6)
      acd83(49)=acd83(9)*acd83(10)
      acd83(48)=acd83(11)+acd83(49)+acd83(48)
      acd83(49)=acd83(3)*acd83(7)
      acd83(50)=acd83(13)*acd83(15)
      acd83(49)=acd83(49)+acd83(50)
      acd83(48)=acd83(49)*acd83(48)
      acd83(49)=acd83(33)*acd83(26)
      acd83(50)=-acd83(34)*acd83(35)
      acd83(49)=acd83(36)+acd83(50)+acd83(49)
      acd83(50)=acd83(24)*acd83(32)
      acd83(51)=acd83(31)*acd83(27)
      acd83(50)=acd83(50)+acd83(51)
      acd83(49)=acd83(50)*acd83(49)
      acd83(50)=acd83(20)*acd83(15)
      acd83(51)=acd83(22)*acd83(7)
      acd83(50)=acd83(51)+acd83(50)
      acd83(50)=acd83(21)*acd83(50)
      acd83(51)=acd83(39)*acd83(32)
      acd83(52)=acd83(44)*acd83(27)
      acd83(51)=acd83(52)+acd83(51)
      acd83(51)=acd83(40)*acd83(51)
      acd83(52)=acd83(42)*acd83(32)
      acd83(53)=acd83(45)*acd83(27)
      acd83(52)=acd83(53)+acd83(52)
      acd83(52)=acd83(43)*acd83(52)
      acd83(53)=acd83(25)*acd83(24)
      acd83(54)=acd83(26)*acd83(53)
      acd83(55)=acd83(28)*acd83(26)
      acd83(56)=acd83(27)*acd83(55)
      acd83(54)=acd83(54)+acd83(56)
      acd83(54)=acd83(23)*acd83(54)
      acd83(56)=acd83(25)*acd83(31)
      acd83(57)=acd83(26)*acd83(56)
      acd83(55)=acd83(32)*acd83(55)
      acd83(55)=acd83(57)+acd83(55)
      acd83(55)=acd83(30)*acd83(55)
      acd83(56)=-acd83(35)*acd83(56)
      acd83(57)=acd83(28)*acd83(35)
      acd83(58)=-acd83(32)*acd83(57)
      acd83(56)=acd83(56)+acd83(58)
      acd83(56)=acd83(37)*acd83(56)
      acd83(53)=-acd83(35)*acd83(53)
      acd83(57)=-acd83(27)*acd83(57)
      acd83(53)=acd83(53)+acd83(57)
      acd83(53)=acd83(38)*acd83(53)
      acd83(57)=acd83(4)*acd83(15)
      acd83(58)=acd83(14)*acd83(7)
      acd83(57)=acd83(57)+acd83(58)
      acd83(57)=acd83(18)*acd83(57)
      acd83(58)=acd83(12)*acd83(15)
      acd83(59)=acd83(16)*acd83(7)
      acd83(58)=acd83(58)+acd83(59)
      acd83(58)=acd83(19)*acd83(58)
      acd83(59)=acd83(23)*acd83(27)
      acd83(60)=acd83(30)*acd83(32)
      acd83(59)=acd83(59)+acd83(60)
      acd83(59)=acd83(29)*acd83(59)
      acd83(60)=acd83(37)*acd83(32)
      acd83(61)=acd83(38)*acd83(27)
      acd83(60)=acd83(60)+acd83(61)
      acd83(60)=acd83(41)*acd83(60)
      acd83(61)=acd83(2)*acd83(1)
      brack=acd83(46)+acd83(47)+acd83(48)+acd83(49)+acd83(50)+acd83(51)+acd83(5&
      &2)+acd83(53)+acd83(54)+acd83(55)+acd83(56)+acd83(57)+acd83(58)+acd83(59)&
      &+acd83(60)+2.0_ki*acd83(61)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd83h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd83
      complex(ki) :: brack
      acd83(1)=spvae1k2(iv1)
      acd83(2)=spval3e2(iv2)
      acd83(3)=spvae2e1(iv3)
      acd83(4)=abb83(28)
      acd83(5)=spval3e2(iv3)
      acd83(6)=spvae2e1(iv2)
      acd83(7)=spval4e2(iv3)
      acd83(8)=abb83(32)
      acd83(9)=spval4e2(iv2)
      acd83(10)=spvae1k2(iv2)
      acd83(11)=spval3e2(iv1)
      acd83(12)=spvae2e1(iv1)
      acd83(13)=spval4e2(iv1)
      acd83(14)=spvae1k2(iv3)
      acd83(15)=spvae2k2(iv1)
      acd83(16)=spval5e1(iv2)
      acd83(17)=spvae1e2(iv3)
      acd83(18)=abb83(9)
      acd83(19)=spval5e1(iv3)
      acd83(20)=spvae1e2(iv2)
      acd83(21)=spvae2k2(iv2)
      acd83(22)=spval5e1(iv1)
      acd83(23)=spvae1e2(iv1)
      acd83(24)=spvae2k2(iv3)
      acd83(25)=spvae2l3(iv3)
      acd83(26)=abb83(31)
      acd83(27)=spvae2l3(iv2)
      acd83(28)=spvae2l3(iv1)
      acd83(29)=acd83(3)*acd83(1)
      acd83(30)=acd83(14)*acd83(12)
      acd83(29)=acd83(29)+acd83(30)
      acd83(30)=acd83(2)*acd83(29)
      acd83(31)=acd83(6)*acd83(1)
      acd83(32)=acd83(12)*acd83(10)
      acd83(31)=acd83(31)+acd83(32)
      acd83(32)=acd83(5)*acd83(31)
      acd83(33)=acd83(10)*acd83(3)
      acd83(34)=acd83(14)*acd83(6)
      acd83(33)=acd83(33)+acd83(34)
      acd83(34)=acd83(11)*acd83(33)
      acd83(30)=acd83(34)+acd83(30)+acd83(32)
      acd83(30)=acd83(4)*acd83(30)
      acd83(31)=acd83(7)*acd83(31)
      acd83(29)=acd83(9)*acd83(29)
      acd83(32)=acd83(13)*acd83(33)
      acd83(29)=acd83(32)+acd83(29)+acd83(31)
      acd83(29)=acd83(8)*acd83(29)
      acd83(31)=acd83(17)*acd83(16)
      acd83(32)=acd83(20)*acd83(19)
      acd83(31)=acd83(31)+acd83(32)
      acd83(32)=acd83(15)*acd83(31)
      acd83(33)=acd83(22)*acd83(17)
      acd83(34)=acd83(23)*acd83(19)
      acd83(33)=acd83(33)+acd83(34)
      acd83(34)=acd83(21)*acd83(33)
      acd83(35)=acd83(22)*acd83(20)
      acd83(36)=acd83(23)*acd83(16)
      acd83(35)=acd83(35)+acd83(36)
      acd83(36)=acd83(24)*acd83(35)
      acd83(32)=acd83(36)+acd83(34)+acd83(32)
      acd83(32)=acd83(18)*acd83(32)
      acd83(34)=-acd83(25)*acd83(35)
      acd83(33)=-acd83(27)*acd83(33)
      acd83(31)=-acd83(28)*acd83(31)
      acd83(31)=acd83(31)+acd83(33)+acd83(34)
      acd83(31)=acd83(26)*acd83(31)
      brack=acd83(29)+acd83(30)+acd83(31)+acd83(32)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd83h0
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
      qshift = -k2+k5
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
end module     p2_gg_httbar_d83h0l1d
