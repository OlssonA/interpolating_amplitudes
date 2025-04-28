module     p2_gg_httbar_d35h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d35h12l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd35h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(95) :: acd35
      complex(ki) :: brack
      acd35(1)=dotproduct(k2,qshift)
      acd35(2)=dotproduct(e1,qshift)
      acd35(3)=abb35(17)
      acd35(4)=abb35(16)
      acd35(5)=dotproduct(l5,qshift)
      acd35(6)=abb35(12)
      acd35(7)=dotproduct(qshift,spvak2l5)
      acd35(8)=abb35(10)
      acd35(9)=dotproduct(qshift,spval3k2)
      acd35(10)=abb35(35)
      acd35(11)=dotproduct(qshift,spval3l5)
      acd35(12)=abb35(41)
      acd35(13)=dotproduct(qshift,spval5l3)
      acd35(14)=abb35(27)
      acd35(15)=dotproduct(qshift,spval5l4)
      acd35(16)=abb35(25)
      acd35(17)=dotproduct(qshift,spvak2e2)
      acd35(18)=abb35(28)
      acd35(19)=dotproduct(qshift,spval3e2)
      acd35(20)=abb35(47)
      acd35(21)=dotproduct(qshift,spvae2l3)
      acd35(22)=abb35(46)
      acd35(23)=dotproduct(qshift,spvae2l4)
      acd35(24)=abb35(38)
      acd35(25)=abb35(9)
      acd35(26)=dotproduct(qshift,qshift)
      acd35(27)=abb35(37)
      acd35(28)=abb35(39)
      acd35(29)=abb35(42)
      acd35(30)=abb35(36)
      acd35(31)=abb35(26)
      acd35(32)=abb35(22)
      acd35(33)=abb35(31)
      acd35(34)=dotproduct(qshift,spvak2l3)
      acd35(35)=abb35(13)
      acd35(36)=dotproduct(qshift,spvak2l4)
      acd35(37)=abb35(20)
      acd35(38)=dotproduct(qshift,spval5k2)
      acd35(39)=abb35(33)
      acd35(40)=dotproduct(qshift,spvak1e1)
      acd35(41)=abb35(14)
      acd35(42)=dotproduct(qshift,spvae1k1)
      acd35(43)=abb35(23)
      acd35(44)=dotproduct(qshift,spvak2e1)
      acd35(45)=abb35(76)
      acd35(46)=dotproduct(qshift,spvae1k2)
      acd35(47)=abb35(32)
      acd35(48)=dotproduct(qshift,spvae2k2)
      acd35(49)=abb35(21)
      acd35(50)=dotproduct(qshift,spval3e1)
      acd35(51)=abb35(49)
      acd35(52)=dotproduct(qshift,spvae1l3)
      acd35(53)=abb35(11)
      acd35(54)=dotproduct(qshift,spvae1l4)
      acd35(55)=abb35(40)
      acd35(56)=dotproduct(qshift,spval5e1)
      acd35(57)=abb35(34)
      acd35(58)=dotproduct(qshift,spvae1l5)
      acd35(59)=abb35(24)
      acd35(60)=dotproduct(qshift,spval5e2)
      acd35(61)=abb35(30)
      acd35(62)=dotproduct(qshift,spvae2l5)
      acd35(63)=abb35(29)
      acd35(64)=dotproduct(qshift,spvae1e2)
      acd35(65)=abb35(19)
      acd35(66)=dotproduct(qshift,spvae2e1)
      acd35(67)=abb35(18)
      acd35(68)=abb35(15)
      acd35(69)=acd35(3)*acd35(1)
      acd35(70)=acd35(8)*acd35(7)
      acd35(71)=acd35(10)*acd35(9)
      acd35(72)=acd35(12)*acd35(11)
      acd35(73)=acd35(14)*acd35(13)
      acd35(74)=acd35(16)*acd35(15)
      acd35(75)=acd35(18)*acd35(17)
      acd35(76)=acd35(20)*acd35(19)
      acd35(77)=acd35(22)*acd35(21)
      acd35(78)=acd35(24)*acd35(23)
      acd35(69)=-acd35(25)+acd35(78)+acd35(77)+acd35(76)+acd35(75)+acd35(74)+ac&
      &d35(73)+acd35(72)+acd35(71)+acd35(70)+acd35(69)
      acd35(69)=acd35(2)*acd35(69)
      acd35(70)=-acd35(4)*acd35(1)
      acd35(71)=-acd35(6)*acd35(5)
      acd35(72)=-acd35(27)*acd35(26)
      acd35(73)=-acd35(28)*acd35(7)
      acd35(74)=-acd35(29)*acd35(9)
      acd35(75)=-acd35(30)*acd35(11)
      acd35(76)=-acd35(31)*acd35(13)
      acd35(77)=-acd35(32)*acd35(15)
      acd35(78)=-acd35(33)*acd35(17)
      acd35(79)=-acd35(35)*acd35(34)
      acd35(80)=-acd35(37)*acd35(36)
      acd35(81)=-acd35(39)*acd35(38)
      acd35(82)=-acd35(41)*acd35(40)
      acd35(83)=-acd35(43)*acd35(42)
      acd35(84)=acd35(45)*acd35(44)
      acd35(85)=-acd35(47)*acd35(46)
      acd35(86)=-acd35(49)*acd35(48)
      acd35(87)=-acd35(51)*acd35(50)
      acd35(88)=-acd35(53)*acd35(52)
      acd35(89)=-acd35(55)*acd35(54)
      acd35(90)=-acd35(57)*acd35(56)
      acd35(91)=-acd35(59)*acd35(58)
      acd35(92)=-acd35(61)*acd35(60)
      acd35(93)=-acd35(63)*acd35(62)
      acd35(94)=-acd35(65)*acd35(64)
      acd35(95)=-acd35(67)*acd35(66)
      brack=acd35(68)+acd35(69)+acd35(70)+acd35(71)+acd35(72)+acd35(73)+acd35(7&
      &4)+acd35(75)+acd35(76)+acd35(77)+acd35(78)+acd35(79)+acd35(80)+acd35(81)&
      &+acd35(82)+acd35(83)+acd35(84)+acd35(85)+acd35(86)+acd35(87)+acd35(88)+a&
      &cd35(89)+acd35(90)+acd35(91)+acd35(92)+acd35(93)+acd35(94)+acd35(95)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd35h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(106) :: acd35
      complex(ki) :: brack
      acd35(1)=k2(iv1)
      acd35(2)=dotproduct(e1,qshift)
      acd35(3)=abb35(17)
      acd35(4)=abb35(16)
      acd35(5)=l5(iv1)
      acd35(6)=abb35(12)
      acd35(7)=e1(iv1)
      acd35(8)=dotproduct(k2,qshift)
      acd35(9)=dotproduct(qshift,spvak2l5)
      acd35(10)=abb35(10)
      acd35(11)=dotproduct(qshift,spval3k2)
      acd35(12)=abb35(35)
      acd35(13)=dotproduct(qshift,spval3l5)
      acd35(14)=abb35(41)
      acd35(15)=dotproduct(qshift,spval5l3)
      acd35(16)=abb35(27)
      acd35(17)=dotproduct(qshift,spval5l4)
      acd35(18)=abb35(25)
      acd35(19)=dotproduct(qshift,spvak2e2)
      acd35(20)=abb35(28)
      acd35(21)=dotproduct(qshift,spval3e2)
      acd35(22)=abb35(47)
      acd35(23)=dotproduct(qshift,spvae2l3)
      acd35(24)=abb35(46)
      acd35(25)=dotproduct(qshift,spvae2l4)
      acd35(26)=abb35(38)
      acd35(27)=abb35(9)
      acd35(28)=qshift(iv1)
      acd35(29)=abb35(37)
      acd35(30)=spvak2l5(iv1)
      acd35(31)=abb35(39)
      acd35(32)=spval3k2(iv1)
      acd35(33)=abb35(42)
      acd35(34)=spval3l5(iv1)
      acd35(35)=abb35(36)
      acd35(36)=spval5l3(iv1)
      acd35(37)=abb35(26)
      acd35(38)=spval5l4(iv1)
      acd35(39)=abb35(22)
      acd35(40)=spvak2e2(iv1)
      acd35(41)=abb35(31)
      acd35(42)=spval3e2(iv1)
      acd35(43)=spvae2l3(iv1)
      acd35(44)=spvae2l4(iv1)
      acd35(45)=spvak2l3(iv1)
      acd35(46)=abb35(13)
      acd35(47)=spvak2l4(iv1)
      acd35(48)=abb35(20)
      acd35(49)=spval5k2(iv1)
      acd35(50)=abb35(33)
      acd35(51)=spvak1e1(iv1)
      acd35(52)=abb35(14)
      acd35(53)=spvae1k1(iv1)
      acd35(54)=abb35(23)
      acd35(55)=spvak2e1(iv1)
      acd35(56)=abb35(76)
      acd35(57)=spvae1k2(iv1)
      acd35(58)=abb35(32)
      acd35(59)=spvae2k2(iv1)
      acd35(60)=abb35(21)
      acd35(61)=spval3e1(iv1)
      acd35(62)=abb35(49)
      acd35(63)=spvae1l3(iv1)
      acd35(64)=abb35(11)
      acd35(65)=spvae1l4(iv1)
      acd35(66)=abb35(40)
      acd35(67)=spval5e1(iv1)
      acd35(68)=abb35(34)
      acd35(69)=spvae1l5(iv1)
      acd35(70)=abb35(24)
      acd35(71)=spval5e2(iv1)
      acd35(72)=abb35(30)
      acd35(73)=spvae2l5(iv1)
      acd35(74)=abb35(29)
      acd35(75)=spvae1e2(iv1)
      acd35(76)=abb35(19)
      acd35(77)=spvae2e1(iv1)
      acd35(78)=abb35(18)
      acd35(79)=-acd35(3)*acd35(1)
      acd35(80)=-acd35(30)*acd35(10)
      acd35(81)=-acd35(32)*acd35(12)
      acd35(82)=-acd35(34)*acd35(14)
      acd35(83)=-acd35(36)*acd35(16)
      acd35(84)=-acd35(38)*acd35(18)
      acd35(85)=-acd35(40)*acd35(20)
      acd35(86)=-acd35(42)*acd35(22)
      acd35(87)=-acd35(43)*acd35(24)
      acd35(88)=-acd35(44)*acd35(26)
      acd35(79)=acd35(88)+acd35(87)+acd35(86)+acd35(85)+acd35(84)+acd35(83)+acd&
      &35(82)+acd35(81)+acd35(79)+acd35(80)
      acd35(79)=acd35(2)*acd35(79)
      acd35(80)=-acd35(8)*acd35(3)
      acd35(81)=-acd35(9)*acd35(10)
      acd35(82)=-acd35(11)*acd35(12)
      acd35(83)=-acd35(13)*acd35(14)
      acd35(84)=-acd35(15)*acd35(16)
      acd35(85)=-acd35(17)*acd35(18)
      acd35(86)=-acd35(19)*acd35(20)
      acd35(87)=-acd35(21)*acd35(22)
      acd35(88)=-acd35(23)*acd35(24)
      acd35(89)=-acd35(25)*acd35(26)
      acd35(80)=acd35(27)+acd35(89)+acd35(88)+acd35(87)+acd35(86)+acd35(85)+acd&
      &35(84)+acd35(83)+acd35(82)+acd35(81)+acd35(80)
      acd35(80)=acd35(7)*acd35(80)
      acd35(81)=acd35(4)*acd35(1)
      acd35(82)=acd35(6)*acd35(5)
      acd35(83)=acd35(29)*acd35(28)
      acd35(84)=acd35(31)*acd35(30)
      acd35(85)=acd35(33)*acd35(32)
      acd35(86)=acd35(35)*acd35(34)
      acd35(87)=acd35(37)*acd35(36)
      acd35(88)=acd35(39)*acd35(38)
      acd35(89)=acd35(41)*acd35(40)
      acd35(90)=acd35(46)*acd35(45)
      acd35(91)=acd35(48)*acd35(47)
      acd35(92)=acd35(50)*acd35(49)
      acd35(93)=acd35(52)*acd35(51)
      acd35(94)=acd35(54)*acd35(53)
      acd35(95)=-acd35(56)*acd35(55)
      acd35(96)=acd35(58)*acd35(57)
      acd35(97)=acd35(60)*acd35(59)
      acd35(98)=acd35(62)*acd35(61)
      acd35(99)=acd35(64)*acd35(63)
      acd35(100)=acd35(66)*acd35(65)
      acd35(101)=acd35(68)*acd35(67)
      acd35(102)=acd35(70)*acd35(69)
      acd35(103)=acd35(72)*acd35(71)
      acd35(104)=acd35(74)*acd35(73)
      acd35(105)=acd35(76)*acd35(75)
      acd35(106)=acd35(78)*acd35(77)
      brack=acd35(79)+acd35(80)+acd35(81)+acd35(82)+2.0_ki*acd35(83)+acd35(84)+&
      &acd35(85)+acd35(86)+acd35(87)+acd35(88)+acd35(89)+acd35(90)+acd35(91)+ac&
      &d35(92)+acd35(93)+acd35(94)+acd35(95)+acd35(96)+acd35(97)+acd35(98)+acd3&
      &5(99)+acd35(100)+acd35(101)+acd35(102)+acd35(103)+acd35(104)+acd35(105)+&
      &acd35(106)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd35h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd35
      complex(ki) :: brack
      acd35(1)=d(iv1,iv2)
      acd35(2)=abb35(37)
      acd35(3)=k2(iv1)
      acd35(4)=e1(iv2)
      acd35(5)=abb35(17)
      acd35(6)=k2(iv2)
      acd35(7)=e1(iv1)
      acd35(8)=spvak2l5(iv2)
      acd35(9)=abb35(10)
      acd35(10)=spval3k2(iv2)
      acd35(11)=abb35(35)
      acd35(12)=spval3l5(iv2)
      acd35(13)=abb35(41)
      acd35(14)=spval5l3(iv2)
      acd35(15)=abb35(27)
      acd35(16)=spval5l4(iv2)
      acd35(17)=abb35(25)
      acd35(18)=spvak2e2(iv2)
      acd35(19)=abb35(28)
      acd35(20)=spval3e2(iv2)
      acd35(21)=abb35(47)
      acd35(22)=spvae2l3(iv2)
      acd35(23)=abb35(46)
      acd35(24)=spvae2l4(iv2)
      acd35(25)=abb35(38)
      acd35(26)=spvak2l5(iv1)
      acd35(27)=spval3k2(iv1)
      acd35(28)=spval3l5(iv1)
      acd35(29)=spval5l3(iv1)
      acd35(30)=spval5l4(iv1)
      acd35(31)=spvak2e2(iv1)
      acd35(32)=spval3e2(iv1)
      acd35(33)=spvae2l3(iv1)
      acd35(34)=spvae2l4(iv1)
      acd35(35)=acd35(3)*acd35(5)
      acd35(36)=acd35(26)*acd35(9)
      acd35(37)=acd35(27)*acd35(11)
      acd35(38)=acd35(28)*acd35(13)
      acd35(39)=acd35(29)*acd35(15)
      acd35(40)=acd35(30)*acd35(17)
      acd35(41)=acd35(31)*acd35(19)
      acd35(42)=acd35(32)*acd35(21)
      acd35(43)=acd35(33)*acd35(23)
      acd35(44)=acd35(34)*acd35(25)
      acd35(35)=acd35(44)+acd35(43)+acd35(42)+acd35(41)+acd35(40)+acd35(39)+acd&
      &35(38)+acd35(37)+acd35(36)+acd35(35)
      acd35(35)=acd35(4)*acd35(35)
      acd35(36)=acd35(6)*acd35(5)
      acd35(37)=acd35(8)*acd35(9)
      acd35(38)=acd35(10)*acd35(11)
      acd35(39)=acd35(12)*acd35(13)
      acd35(40)=acd35(14)*acd35(15)
      acd35(41)=acd35(16)*acd35(17)
      acd35(42)=acd35(18)*acd35(19)
      acd35(43)=acd35(20)*acd35(21)
      acd35(44)=acd35(22)*acd35(23)
      acd35(45)=acd35(24)*acd35(25)
      acd35(36)=acd35(45)+acd35(44)+acd35(43)+acd35(42)+acd35(41)+acd35(40)+acd&
      &35(39)+acd35(38)+acd35(37)+acd35(36)
      acd35(36)=acd35(7)*acd35(36)
      acd35(37)=acd35(2)*acd35(1)
      brack=acd35(35)+acd35(36)-2.0_ki*acd35(37)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd35h12_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k2
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
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d35h12l1d_qp
