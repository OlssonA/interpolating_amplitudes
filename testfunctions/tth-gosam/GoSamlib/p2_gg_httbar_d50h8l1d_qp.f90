module     p2_gg_httbar_d50h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d50h8l1d_qp.f90
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
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(43) :: acd50
      complex(ki) :: brack
      acd50(1)=dotproduct(k2,qshift)
      acd50(2)=abb50(10)
      acd50(3)=dotproduct(qshift,spvak2l3)
      acd50(4)=abb50(24)
      acd50(5)=dotproduct(qshift,spval3k2)
      acd50(6)=abb50(18)
      acd50(7)=abb50(13)
      acd50(8)=dotproduct(qshift,spval4k2)
      acd50(9)=abb50(29)
      acd50(10)=abb50(19)
      acd50(11)=dotproduct(qshift,spvak2l5)
      acd50(12)=abb50(37)
      acd50(13)=abb50(15)
      acd50(14)=dotproduct(qshift,spvak1k2)
      acd50(15)=dotproduct(qshift,spvak2k1)
      acd50(16)=abb50(9)
      acd50(17)=dotproduct(qshift,spval3k1)
      acd50(18)=abb50(39)
      acd50(19)=abb50(26)
      acd50(20)=dotproduct(qshift,spvak1l3)
      acd50(21)=abb50(32)
      acd50(22)=abb50(25)
      acd50(23)=dotproduct(qshift,spvak1l5)
      acd50(24)=abb50(14)
      acd50(25)=dotproduct(qshift,spval4k1)
      acd50(26)=abb50(23)
      acd50(27)=abb50(17)
      acd50(28)=abb50(12)
      acd50(29)=abb50(21)
      acd50(30)=abb50(11)
      acd50(31)=abb50(20)
      acd50(32)=abb50(22)
      acd50(33)=acd50(5)*acd50(6)
      acd50(34)=acd50(3)*acd50(4)
      acd50(35)=acd50(1)*acd50(2)
      acd50(33)=acd50(35)+acd50(34)-acd50(7)+acd50(33)
      acd50(33)=acd50(1)*acd50(33)
      acd50(34)=-acd50(17)*acd50(18)
      acd50(35)=-acd50(15)*acd50(16)
      acd50(34)=acd50(35)-acd50(19)+acd50(34)
      acd50(34)=acd50(14)*acd50(34)
      acd50(35)=-acd50(25)*acd50(28)
      acd50(36)=acd50(25)*acd50(27)
      acd50(36)=-acd50(29)+acd50(36)
      acd50(36)=acd50(23)*acd50(36)
      acd50(37)=acd50(25)*acd50(9)
      acd50(37)=-acd50(26)+acd50(37)
      acd50(37)=acd50(20)*acd50(37)
      acd50(38)=acd50(23)*acd50(12)
      acd50(38)=-acd50(24)+acd50(38)
      acd50(38)=acd50(17)*acd50(38)
      acd50(39)=-acd50(20)*acd50(21)
      acd50(39)=-acd50(22)+acd50(39)
      acd50(39)=acd50(15)*acd50(39)
      acd50(40)=-acd50(11)*acd50(31)
      acd50(41)=-acd50(11)*acd50(27)
      acd50(41)=-acd50(30)+acd50(41)
      acd50(41)=acd50(8)*acd50(41)
      acd50(42)=-acd50(11)*acd50(12)
      acd50(42)=-acd50(13)+acd50(42)
      acd50(42)=acd50(5)*acd50(42)
      acd50(43)=-acd50(8)*acd50(9)
      acd50(43)=-acd50(10)+acd50(43)
      acd50(43)=acd50(3)*acd50(43)
      brack=acd50(32)+acd50(33)+acd50(34)+acd50(35)+acd50(36)+acd50(37)+acd50(3&
      &8)+acd50(39)+acd50(40)+acd50(41)+acd50(42)+acd50(43)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(57) :: acd50
      complex(ki) :: brack
      acd50(1)=k2(iv1)
      acd50(2)=dotproduct(k2,qshift)
      acd50(3)=abb50(10)
      acd50(4)=dotproduct(qshift,spvak2l3)
      acd50(5)=abb50(24)
      acd50(6)=dotproduct(qshift,spval3k2)
      acd50(7)=abb50(18)
      acd50(8)=abb50(13)
      acd50(9)=spvak2l3(iv1)
      acd50(10)=dotproduct(qshift,spval4k2)
      acd50(11)=abb50(29)
      acd50(12)=abb50(19)
      acd50(13)=spval3k2(iv1)
      acd50(14)=dotproduct(qshift,spvak2l5)
      acd50(15)=abb50(37)
      acd50(16)=abb50(15)
      acd50(17)=spvak1k2(iv1)
      acd50(18)=dotproduct(qshift,spvak2k1)
      acd50(19)=abb50(9)
      acd50(20)=dotproduct(qshift,spval3k1)
      acd50(21)=abb50(39)
      acd50(22)=abb50(26)
      acd50(23)=spvak2k1(iv1)
      acd50(24)=dotproduct(qshift,spvak1k2)
      acd50(25)=dotproduct(qshift,spvak1l3)
      acd50(26)=abb50(32)
      acd50(27)=abb50(25)
      acd50(28)=spval3k1(iv1)
      acd50(29)=dotproduct(qshift,spvak1l5)
      acd50(30)=abb50(14)
      acd50(31)=spvak1l3(iv1)
      acd50(32)=dotproduct(qshift,spval4k1)
      acd50(33)=abb50(23)
      acd50(34)=spval4k1(iv1)
      acd50(35)=abb50(17)
      acd50(36)=abb50(12)
      acd50(37)=spvak1l5(iv1)
      acd50(38)=abb50(21)
      acd50(39)=spval4k2(iv1)
      acd50(40)=abb50(11)
      acd50(41)=spvak2l5(iv1)
      acd50(42)=abb50(20)
      acd50(43)=acd50(41)*acd50(10)
      acd50(44)=acd50(39)*acd50(14)
      acd50(45)=-acd50(37)*acd50(32)
      acd50(46)=-acd50(34)*acd50(29)
      acd50(43)=acd50(46)+acd50(45)+acd50(43)+acd50(44)
      acd50(43)=acd50(35)*acd50(43)
      acd50(44)=acd50(41)*acd50(6)
      acd50(45)=-acd50(37)*acd50(20)
      acd50(46)=-acd50(28)*acd50(29)
      acd50(47)=acd50(13)*acd50(14)
      acd50(44)=acd50(47)+acd50(46)+acd50(44)+acd50(45)
      acd50(44)=acd50(15)*acd50(44)
      acd50(45)=acd50(39)*acd50(4)
      acd50(46)=-acd50(34)*acd50(25)
      acd50(47)=-acd50(31)*acd50(32)
      acd50(48)=acd50(9)*acd50(10)
      acd50(45)=acd50(48)+acd50(47)+acd50(45)+acd50(46)
      acd50(45)=acd50(11)*acd50(45)
      acd50(46)=-acd50(6)*acd50(7)
      acd50(47)=-acd50(4)*acd50(5)
      acd50(48)=acd50(2)*acd50(3)
      acd50(46)=-2.0_ki*acd50(48)+acd50(47)+acd50(8)+acd50(46)
      acd50(46)=acd50(1)*acd50(46)
      acd50(47)=acd50(25)*acd50(26)
      acd50(48)=acd50(19)*acd50(24)
      acd50(47)=acd50(48)+acd50(27)+acd50(47)
      acd50(47)=acd50(23)*acd50(47)
      acd50(48)=acd50(20)*acd50(21)
      acd50(49)=acd50(18)*acd50(19)
      acd50(48)=acd50(49)+acd50(22)+acd50(48)
      acd50(48)=acd50(17)*acd50(48)
      acd50(49)=-acd50(13)*acd50(7)
      acd50(50)=-acd50(9)*acd50(5)
      acd50(49)=acd50(49)+acd50(50)
      acd50(49)=acd50(2)*acd50(49)
      acd50(50)=acd50(41)*acd50(42)
      acd50(51)=acd50(39)*acd50(40)
      acd50(52)=acd50(37)*acd50(38)
      acd50(53)=acd50(34)*acd50(36)
      acd50(54)=acd50(18)*acd50(26)
      acd50(54)=acd50(33)+acd50(54)
      acd50(54)=acd50(31)*acd50(54)
      acd50(55)=acd50(21)*acd50(24)
      acd50(55)=acd50(30)+acd50(55)
      acd50(55)=acd50(28)*acd50(55)
      acd50(56)=acd50(13)*acd50(16)
      acd50(57)=acd50(9)*acd50(12)
      brack=acd50(43)+acd50(44)+acd50(45)+acd50(46)+acd50(47)+acd50(48)+acd50(4&
      &9)+acd50(50)+acd50(51)+acd50(52)+acd50(53)+acd50(54)+acd50(55)+acd50(56)&
      &+acd50(57)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd50h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd50
      complex(ki) :: brack
      acd50(1)=k2(iv1)
      acd50(2)=k2(iv2)
      acd50(3)=abb50(10)
      acd50(4)=spvak2l3(iv2)
      acd50(5)=abb50(24)
      acd50(6)=spval3k2(iv2)
      acd50(7)=abb50(18)
      acd50(8)=spvak2l3(iv1)
      acd50(9)=spval3k2(iv1)
      acd50(10)=spval4k2(iv2)
      acd50(11)=abb50(29)
      acd50(12)=spval4k2(iv1)
      acd50(13)=spvak2l5(iv2)
      acd50(14)=abb50(37)
      acd50(15)=spvak2l5(iv1)
      acd50(16)=spvak1k2(iv1)
      acd50(17)=spvak2k1(iv2)
      acd50(18)=abb50(9)
      acd50(19)=spval3k1(iv2)
      acd50(20)=abb50(39)
      acd50(21)=spvak1k2(iv2)
      acd50(22)=spvak2k1(iv1)
      acd50(23)=spval3k1(iv1)
      acd50(24)=spvak1l3(iv2)
      acd50(25)=abb50(32)
      acd50(26)=spvak1l3(iv1)
      acd50(27)=spvak1l5(iv2)
      acd50(28)=spvak1l5(iv1)
      acd50(29)=spval4k1(iv2)
      acd50(30)=spval4k1(iv1)
      acd50(31)=abb50(17)
      acd50(32)=acd50(28)*acd50(29)
      acd50(33)=acd50(27)*acd50(30)
      acd50(34)=-acd50(12)*acd50(13)
      acd50(35)=-acd50(10)*acd50(15)
      acd50(32)=acd50(35)+acd50(34)+acd50(32)+acd50(33)
      acd50(32)=acd50(31)*acd50(32)
      acd50(33)=acd50(23)*acd50(27)
      acd50(34)=acd50(19)*acd50(28)
      acd50(35)=-acd50(9)*acd50(13)
      acd50(36)=-acd50(6)*acd50(15)
      acd50(33)=acd50(36)+acd50(35)+acd50(33)+acd50(34)
      acd50(33)=acd50(14)*acd50(33)
      acd50(34)=acd50(26)*acd50(29)
      acd50(35)=acd50(24)*acd50(30)
      acd50(36)=-acd50(8)*acd50(10)
      acd50(37)=-acd50(4)*acd50(12)
      acd50(34)=acd50(37)+acd50(36)+acd50(34)+acd50(35)
      acd50(34)=acd50(11)*acd50(34)
      acd50(35)=acd50(6)*acd50(7)
      acd50(36)=acd50(4)*acd50(5)
      acd50(37)=acd50(2)*acd50(3)
      acd50(35)=2.0_ki*acd50(37)+acd50(35)+acd50(36)
      acd50(35)=acd50(1)*acd50(35)
      acd50(36)=-acd50(22)*acd50(24)
      acd50(37)=-acd50(17)*acd50(26)
      acd50(36)=acd50(37)+acd50(36)
      acd50(36)=acd50(25)*acd50(36)
      acd50(37)=-acd50(20)*acd50(23)
      acd50(38)=-acd50(18)*acd50(22)
      acd50(37)=acd50(38)+acd50(37)
      acd50(37)=acd50(21)*acd50(37)
      acd50(38)=-acd50(19)*acd50(20)
      acd50(39)=-acd50(17)*acd50(18)
      acd50(38)=acd50(38)+acd50(39)
      acd50(38)=acd50(16)*acd50(38)
      acd50(39)=acd50(7)*acd50(9)
      acd50(40)=acd50(5)*acd50(8)
      acd50(39)=acd50(39)+acd50(40)
      acd50(39)=acd50(2)*acd50(39)
      brack=acd50(32)+acd50(33)+acd50(34)+acd50(35)+acd50(36)+acd50(37)+acd50(3&
      &8)+acd50(39)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd50h8_qp
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
      qshift = -k3-k5
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
end module     p2_gg_httbar_d50h8l1d_qp
