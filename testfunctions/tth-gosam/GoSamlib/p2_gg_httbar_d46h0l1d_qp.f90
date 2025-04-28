module     p2_gg_httbar_d46h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d46h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd46h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(50) :: acd46
      complex(ki) :: brack
      acd46(1)=dotproduct(k1,qshift)
      acd46(2)=dotproduct(qshift,spval3k2)
      acd46(3)=abb46(19)
      acd46(4)=dotproduct(qshift,spval4k2)
      acd46(5)=abb46(10)
      acd46(6)=dotproduct(qshift,spval4l3)
      acd46(7)=abb46(27)
      acd46(8)=dotproduct(qshift,spval5k2)
      acd46(9)=abb46(24)
      acd46(10)=dotproduct(qshift,spval5l3)
      acd46(11)=abb46(21)
      acd46(12)=abb46(13)
      acd46(13)=dotproduct(k2,qshift)
      acd46(14)=abb46(18)
      acd46(15)=abb46(23)
      acd46(16)=abb46(26)
      acd46(17)=abb46(14)
      acd46(18)=dotproduct(qshift,qshift)
      acd46(19)=abb46(17)
      acd46(20)=abb46(34)
      acd46(21)=dotproduct(qshift,spvak2l3)
      acd46(22)=abb46(38)
      acd46(23)=abb46(28)
      acd46(24)=abb46(37)
      acd46(25)=abb46(35)
      acd46(26)=dotproduct(qshift,spvak1k2)
      acd46(27)=dotproduct(qshift,spval3k1)
      acd46(28)=abb46(16)
      acd46(29)=dotproduct(qshift,spval4k1)
      acd46(30)=abb46(15)
      acd46(31)=dotproduct(qshift,spval5k1)
      acd46(32)=abb46(12)
      acd46(33)=abb46(9)
      acd46(34)=dotproduct(qshift,spvak1l3)
      acd46(35)=abb46(32)
      acd46(36)=abb46(36)
      acd46(37)=abb46(11)
      acd46(38)=abb46(25)
      acd46(39)=abb46(43)
      acd46(40)=acd46(10)*acd46(11)
      acd46(41)=acd46(6)*acd46(7)
      acd46(40)=acd46(40)+acd46(41)
      acd46(41)=acd46(2)*acd46(14)
      acd46(42)=acd46(8)*acd46(16)
      acd46(43)=acd46(4)*acd46(15)
      acd46(41)=acd46(43)+acd46(42)+acd46(41)-acd46(17)-acd46(40)
      acd46(41)=acd46(13)*acd46(41)
      acd46(42)=acd46(2)*acd46(3)
      acd46(43)=acd46(8)*acd46(9)
      acd46(44)=acd46(4)*acd46(5)
      acd46(40)=acd46(44)+acd46(43)+acd46(42)-acd46(12)+acd46(40)
      acd46(40)=acd46(1)*acd46(40)
      acd46(42)=acd46(27)*acd46(28)
      acd46(43)=acd46(31)*acd46(32)
      acd46(44)=acd46(29)*acd46(30)
      acd46(42)=acd46(44)+acd46(43)-acd46(33)+acd46(42)
      acd46(42)=acd46(26)*acd46(42)
      acd46(43)=acd46(18)*acd46(19)
      acd46(44)=-acd46(34)*acd46(37)
      acd46(45)=acd46(34)*acd46(24)
      acd46(45)=-acd46(36)+acd46(45)
      acd46(45)=acd46(31)*acd46(45)
      acd46(46)=-acd46(34)*acd46(22)
      acd46(46)=-acd46(35)+acd46(46)
      acd46(46)=acd46(29)*acd46(46)
      acd46(47)=-acd46(21)*acd46(38)
      acd46(48)=-acd46(2)*acd46(20)
      acd46(49)=-acd46(21)*acd46(24)
      acd46(49)=-acd46(25)+acd46(49)
      acd46(49)=acd46(8)*acd46(49)
      acd46(50)=acd46(21)*acd46(22)
      acd46(50)=-acd46(23)+acd46(50)
      acd46(50)=acd46(4)*acd46(50)
      brack=acd46(39)+acd46(40)+acd46(41)+acd46(42)+acd46(43)+acd46(44)+acd46(4&
      &5)+acd46(46)+acd46(47)+acd46(48)+acd46(49)+acd46(50)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd46h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(67) :: acd46
      complex(ki) :: brack
      acd46(1)=k1(iv1)
      acd46(2)=dotproduct(qshift,spval3k2)
      acd46(3)=abb46(19)
      acd46(4)=dotproduct(qshift,spval4k2)
      acd46(5)=abb46(10)
      acd46(6)=dotproduct(qshift,spval4l3)
      acd46(7)=abb46(27)
      acd46(8)=dotproduct(qshift,spval5k2)
      acd46(9)=abb46(24)
      acd46(10)=dotproduct(qshift,spval5l3)
      acd46(11)=abb46(21)
      acd46(12)=abb46(13)
      acd46(13)=k2(iv1)
      acd46(14)=abb46(18)
      acd46(15)=abb46(23)
      acd46(16)=abb46(26)
      acd46(17)=abb46(14)
      acd46(18)=qshift(iv1)
      acd46(19)=abb46(17)
      acd46(20)=spval3k2(iv1)
      acd46(21)=dotproduct(k1,qshift)
      acd46(22)=dotproduct(k2,qshift)
      acd46(23)=abb46(34)
      acd46(24)=spval4k2(iv1)
      acd46(25)=dotproduct(qshift,spvak2l3)
      acd46(26)=abb46(38)
      acd46(27)=abb46(28)
      acd46(28)=spval4l3(iv1)
      acd46(29)=spval5k2(iv1)
      acd46(30)=abb46(37)
      acd46(31)=abb46(35)
      acd46(32)=spval5l3(iv1)
      acd46(33)=spvak1k2(iv1)
      acd46(34)=dotproduct(qshift,spval3k1)
      acd46(35)=abb46(16)
      acd46(36)=dotproduct(qshift,spval4k1)
      acd46(37)=abb46(15)
      acd46(38)=dotproduct(qshift,spval5k1)
      acd46(39)=abb46(12)
      acd46(40)=abb46(9)
      acd46(41)=spval3k1(iv1)
      acd46(42)=dotproduct(qshift,spvak1k2)
      acd46(43)=spval4k1(iv1)
      acd46(44)=dotproduct(qshift,spvak1l3)
      acd46(45)=abb46(32)
      acd46(46)=spval5k1(iv1)
      acd46(47)=abb46(36)
      acd46(48)=spvak1l3(iv1)
      acd46(49)=abb46(11)
      acd46(50)=spvak2l3(iv1)
      acd46(51)=abb46(25)
      acd46(52)=acd46(11)*acd46(32)
      acd46(53)=acd46(7)*acd46(28)
      acd46(52)=acd46(52)+acd46(53)
      acd46(53)=acd46(20)*acd46(14)
      acd46(54)=acd46(29)*acd46(16)
      acd46(55)=acd46(24)*acd46(15)
      acd46(53)=acd46(55)+acd46(53)+acd46(54)-acd46(52)
      acd46(53)=acd46(22)*acd46(53)
      acd46(54)=acd46(20)*acd46(3)
      acd46(55)=acd46(29)*acd46(9)
      acd46(56)=acd46(24)*acd46(5)
      acd46(52)=acd46(56)+acd46(54)+acd46(55)+acd46(52)
      acd46(52)=acd46(21)*acd46(52)
      acd46(54)=acd46(11)*acd46(10)
      acd46(55)=acd46(7)*acd46(6)
      acd46(54)=acd46(54)+acd46(55)
      acd46(55)=acd46(2)*acd46(14)
      acd46(56)=acd46(8)*acd46(16)
      acd46(57)=acd46(4)*acd46(15)
      acd46(55)=acd46(57)+acd46(56)-acd46(17)+acd46(55)-acd46(54)
      acd46(55)=acd46(13)*acd46(55)
      acd46(56)=acd46(2)*acd46(3)
      acd46(57)=acd46(8)*acd46(9)
      acd46(58)=acd46(4)*acd46(5)
      acd46(54)=acd46(58)+acd46(57)-acd46(12)+acd46(56)+acd46(54)
      acd46(54)=acd46(1)*acd46(54)
      acd46(56)=acd46(35)*acd46(41)
      acd46(57)=acd46(46)*acd46(39)
      acd46(58)=acd46(43)*acd46(37)
      acd46(56)=acd46(58)+acd46(56)+acd46(57)
      acd46(56)=acd46(42)*acd46(56)
      acd46(57)=acd46(38)*acd46(39)
      acd46(58)=acd46(36)*acd46(37)
      acd46(59)=acd46(35)*acd46(34)
      acd46(57)=acd46(59)+acd46(58)-acd46(40)+acd46(57)
      acd46(57)=acd46(33)*acd46(57)
      acd46(58)=acd46(48)*acd46(38)
      acd46(59)=acd46(46)*acd46(44)
      acd46(60)=-acd46(8)*acd46(50)
      acd46(58)=acd46(60)+acd46(58)+acd46(59)
      acd46(58)=acd46(30)*acd46(58)
      acd46(59)=-acd46(48)*acd46(36)
      acd46(60)=-acd46(43)*acd46(44)
      acd46(61)=acd46(4)*acd46(50)
      acd46(59)=acd46(61)+acd46(59)+acd46(60)
      acd46(59)=acd46(26)*acd46(59)
      acd46(60)=acd46(18)*acd46(19)
      acd46(61)=-acd46(50)*acd46(51)
      acd46(62)=-acd46(48)*acd46(49)
      acd46(63)=-acd46(46)*acd46(47)
      acd46(64)=-acd46(43)*acd46(45)
      acd46(65)=-acd46(20)*acd46(23)
      acd46(66)=-acd46(30)*acd46(25)
      acd46(66)=-acd46(31)+acd46(66)
      acd46(66)=acd46(29)*acd46(66)
      acd46(67)=acd46(26)*acd46(25)
      acd46(67)=-acd46(27)+acd46(67)
      acd46(67)=acd46(24)*acd46(67)
      brack=acd46(52)+acd46(53)+acd46(54)+acd46(55)+acd46(56)+acd46(57)+acd46(5&
      &8)+acd46(59)+2.0_ki*acd46(60)+acd46(61)+acd46(62)+acd46(63)+acd46(64)+ac&
      &d46(65)+acd46(66)+acd46(67)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd46h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd46
      complex(ki) :: brack
      acd46(1)=d(iv1,iv2)
      acd46(2)=abb46(17)
      acd46(3)=k1(iv1)
      acd46(4)=spval3k2(iv2)
      acd46(5)=abb46(19)
      acd46(6)=spval4k2(iv2)
      acd46(7)=abb46(10)
      acd46(8)=spval4l3(iv2)
      acd46(9)=abb46(27)
      acd46(10)=spval5k2(iv2)
      acd46(11)=abb46(24)
      acd46(12)=spval5l3(iv2)
      acd46(13)=abb46(21)
      acd46(14)=k1(iv2)
      acd46(15)=spval3k2(iv1)
      acd46(16)=spval4k2(iv1)
      acd46(17)=spval4l3(iv1)
      acd46(18)=spval5k2(iv1)
      acd46(19)=spval5l3(iv1)
      acd46(20)=k2(iv1)
      acd46(21)=abb46(18)
      acd46(22)=abb46(23)
      acd46(23)=abb46(26)
      acd46(24)=k2(iv2)
      acd46(25)=spvak2l3(iv2)
      acd46(26)=abb46(38)
      acd46(27)=spvak2l3(iv1)
      acd46(28)=abb46(37)
      acd46(29)=spvak1k2(iv1)
      acd46(30)=spval3k1(iv2)
      acd46(31)=abb46(16)
      acd46(32)=spval4k1(iv2)
      acd46(33)=abb46(15)
      acd46(34)=spval5k1(iv2)
      acd46(35)=abb46(12)
      acd46(36)=spvak1k2(iv2)
      acd46(37)=spval3k1(iv1)
      acd46(38)=spval4k1(iv1)
      acd46(39)=spval5k1(iv1)
      acd46(40)=spvak1l3(iv2)
      acd46(41)=spvak1l3(iv1)
      acd46(42)=acd46(39)*acd46(40)
      acd46(43)=acd46(34)*acd46(41)
      acd46(44)=-acd46(18)*acd46(25)
      acd46(45)=-acd46(10)*acd46(27)
      acd46(42)=acd46(45)+acd46(44)+acd46(42)+acd46(43)
      acd46(42)=acd46(28)*acd46(42)
      acd46(43)=-acd46(38)*acd46(40)
      acd46(44)=-acd46(32)*acd46(41)
      acd46(45)=acd46(16)*acd46(25)
      acd46(46)=acd46(6)*acd46(27)
      acd46(43)=acd46(46)+acd46(45)+acd46(43)+acd46(44)
      acd46(43)=acd46(26)*acd46(43)
      acd46(44)=acd46(13)*acd46(19)
      acd46(45)=acd46(9)*acd46(17)
      acd46(44)=acd46(44)+acd46(45)
      acd46(45)=acd46(15)*acd46(21)
      acd46(46)=acd46(18)*acd46(23)
      acd46(47)=acd46(16)*acd46(22)
      acd46(45)=acd46(47)+acd46(45)+acd46(46)-acd46(44)
      acd46(45)=acd46(24)*acd46(45)
      acd46(46)=acd46(13)*acd46(12)
      acd46(47)=acd46(9)*acd46(8)
      acd46(46)=acd46(46)+acd46(47)
      acd46(47)=acd46(4)*acd46(21)
      acd46(48)=acd46(10)*acd46(23)
      acd46(49)=acd46(6)*acd46(22)
      acd46(47)=acd46(49)+acd46(47)+acd46(48)-acd46(46)
      acd46(47)=acd46(20)*acd46(47)
      acd46(48)=acd46(5)*acd46(15)
      acd46(49)=acd46(18)*acd46(11)
      acd46(50)=acd46(16)*acd46(7)
      acd46(44)=acd46(50)+acd46(48)+acd46(49)+acd46(44)
      acd46(44)=acd46(14)*acd46(44)
      acd46(48)=acd46(4)*acd46(5)
      acd46(49)=acd46(10)*acd46(11)
      acd46(50)=acd46(6)*acd46(7)
      acd46(46)=acd46(50)+acd46(48)+acd46(49)+acd46(46)
      acd46(46)=acd46(3)*acd46(46)
      acd46(48)=acd46(35)*acd46(39)
      acd46(49)=acd46(33)*acd46(38)
      acd46(50)=acd46(31)*acd46(37)
      acd46(48)=acd46(50)+acd46(48)+acd46(49)
      acd46(48)=acd46(36)*acd46(48)
      acd46(49)=acd46(34)*acd46(35)
      acd46(50)=acd46(32)*acd46(33)
      acd46(51)=acd46(31)*acd46(30)
      acd46(49)=acd46(51)+acd46(49)+acd46(50)
      acd46(49)=acd46(29)*acd46(49)
      acd46(50)=acd46(1)*acd46(2)
      brack=acd46(42)+acd46(43)+acd46(44)+acd46(45)+acd46(46)+acd46(47)+acd46(4&
      &8)+acd46(49)+2.0_ki*acd46(50)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd46h0_qp
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
      qshift = -k3-k4-k5
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
end module     p2_gg_httbar_d46h0l1d_qp
