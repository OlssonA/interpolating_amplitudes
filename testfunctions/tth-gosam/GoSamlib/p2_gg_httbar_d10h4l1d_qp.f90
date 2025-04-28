module     p2_gg_httbar_d10h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d10h4l1d_qp.f90
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
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd10
      complex(ki) :: brack
      acd10(1)=dotproduct(k2,qshift)
      acd10(2)=abb10(10)
      acd10(3)=dotproduct(qshift,spval3k2)
      acd10(4)=abb10(30)
      acd10(5)=abb10(11)
      acd10(6)=abb10(16)
      acd10(7)=dotproduct(qshift,spvak1k2)
      acd10(8)=dotproduct(qshift,spvak2k1)
      acd10(9)=abb10(9)
      acd10(10)=dotproduct(qshift,spval3k1)
      acd10(11)=abb10(23)
      acd10(12)=abb10(13)
      acd10(13)=abb10(17)
      acd10(14)=abb10(12)
      acd10(15)=dotproduct(qshift,spvak1l3)
      acd10(16)=dotproduct(qshift,spval5k1)
      acd10(17)=abb10(24)
      acd10(18)=abb10(15)
      acd10(19)=dotproduct(qshift,spvak1l4)
      acd10(20)=abb10(19)
      acd10(21)=abb10(18)
      acd10(22)=abb10(27)
      acd10(23)=dotproduct(qshift,spvak2l3)
      acd10(24)=dotproduct(qshift,spval5k2)
      acd10(25)=abb10(21)
      acd10(26)=dotproduct(qshift,spvak2l4)
      acd10(27)=abb10(14)
      acd10(28)=abb10(22)
      acd10(29)=abb10(20)
      acd10(30)=acd10(20)*acd10(26)
      acd10(31)=acd10(17)*acd10(23)
      acd10(30)=acd10(31)-acd10(27)+acd10(30)
      acd10(30)=acd10(24)*acd10(30)
      acd10(31)=-acd10(19)*acd10(20)
      acd10(32)=-acd10(15)*acd10(17)
      acd10(31)=acd10(32)-acd10(21)+acd10(31)
      acd10(31)=acd10(16)*acd10(31)
      acd10(32)=acd10(10)*acd10(11)
      acd10(33)=acd10(8)*acd10(9)
      acd10(32)=acd10(33)-acd10(12)+acd10(32)
      acd10(32)=acd10(7)*acd10(32)
      acd10(33)=-acd10(3)*acd10(4)
      acd10(34)=acd10(1)*acd10(2)
      acd10(33)=acd10(34)-acd10(5)+acd10(33)
      acd10(33)=acd10(1)*acd10(33)
      acd10(34)=-acd10(26)*acd10(28)
      acd10(35)=-acd10(23)*acd10(25)
      acd10(36)=-acd10(19)*acd10(22)
      acd10(37)=-acd10(15)*acd10(18)
      acd10(38)=-acd10(10)*acd10(14)
      acd10(39)=-acd10(8)*acd10(13)
      acd10(40)=-acd10(3)*acd10(6)
      brack=acd10(29)+acd10(30)+acd10(31)+acd10(32)+acd10(33)+acd10(34)+acd10(3&
      &5)+acd10(36)+acd10(37)+acd10(38)+acd10(39)+acd10(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(52) :: acd10
      complex(ki) :: brack
      acd10(1)=k2(iv1)
      acd10(2)=dotproduct(k2,qshift)
      acd10(3)=abb10(10)
      acd10(4)=dotproduct(qshift,spval3k2)
      acd10(5)=abb10(30)
      acd10(6)=abb10(11)
      acd10(7)=spval3k2(iv1)
      acd10(8)=abb10(16)
      acd10(9)=spvak1k2(iv1)
      acd10(10)=dotproduct(qshift,spvak2k1)
      acd10(11)=abb10(9)
      acd10(12)=dotproduct(qshift,spval3k1)
      acd10(13)=abb10(23)
      acd10(14)=abb10(13)
      acd10(15)=spvak2k1(iv1)
      acd10(16)=dotproduct(qshift,spvak1k2)
      acd10(17)=abb10(17)
      acd10(18)=spval3k1(iv1)
      acd10(19)=abb10(12)
      acd10(20)=spvak1l3(iv1)
      acd10(21)=dotproduct(qshift,spval5k1)
      acd10(22)=abb10(24)
      acd10(23)=abb10(15)
      acd10(24)=spval5k1(iv1)
      acd10(25)=dotproduct(qshift,spvak1l3)
      acd10(26)=dotproduct(qshift,spvak1l4)
      acd10(27)=abb10(19)
      acd10(28)=abb10(18)
      acd10(29)=spvak1l4(iv1)
      acd10(30)=abb10(27)
      acd10(31)=spvak2l3(iv1)
      acd10(32)=dotproduct(qshift,spval5k2)
      acd10(33)=abb10(21)
      acd10(34)=spval5k2(iv1)
      acd10(35)=dotproduct(qshift,spvak2l3)
      acd10(36)=dotproduct(qshift,spvak2l4)
      acd10(37)=abb10(14)
      acd10(38)=spvak2l4(iv1)
      acd10(39)=abb10(22)
      acd10(40)=-acd10(21)*acd10(20)
      acd10(41)=acd10(32)*acd10(31)
      acd10(42)=-acd10(25)*acd10(24)
      acd10(43)=acd10(35)*acd10(34)
      acd10(40)=acd10(43)+acd10(42)+acd10(41)+acd10(40)
      acd10(40)=acd10(22)*acd10(40)
      acd10(41)=-acd10(29)*acd10(21)
      acd10(42)=acd10(38)*acd10(32)
      acd10(43)=-acd10(26)*acd10(24)
      acd10(44)=acd10(36)*acd10(34)
      acd10(41)=acd10(44)+acd10(43)+acd10(42)+acd10(41)
      acd10(41)=acd10(27)*acd10(41)
      acd10(42)=acd10(3)*acd10(2)
      acd10(43)=-acd10(4)*acd10(5)
      acd10(42)=-acd10(6)+acd10(43)+2.0_ki*acd10(42)
      acd10(42)=acd10(1)*acd10(42)
      acd10(43)=acd10(10)*acd10(11)
      acd10(44)=acd10(12)*acd10(13)
      acd10(43)=-acd10(14)+acd10(44)+acd10(43)
      acd10(43)=acd10(9)*acd10(43)
      acd10(44)=-acd10(5)*acd10(2)
      acd10(44)=-acd10(8)+acd10(44)
      acd10(44)=acd10(7)*acd10(44)
      acd10(45)=acd10(16)*acd10(11)
      acd10(45)=-acd10(17)+acd10(45)
      acd10(45)=acd10(15)*acd10(45)
      acd10(46)=acd10(16)*acd10(13)
      acd10(46)=-acd10(19)+acd10(46)
      acd10(46)=acd10(18)*acd10(46)
      acd10(47)=-acd10(23)*acd10(20)
      acd10(48)=-acd10(28)*acd10(24)
      acd10(49)=-acd10(30)*acd10(29)
      acd10(50)=-acd10(33)*acd10(31)
      acd10(51)=-acd10(37)*acd10(34)
      acd10(52)=-acd10(39)*acd10(38)
      brack=acd10(40)+acd10(41)+acd10(42)+acd10(43)+acd10(44)+acd10(45)+acd10(4&
      &6)+acd10(47)+acd10(48)+acd10(49)+acd10(50)+acd10(51)+acd10(52)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd10h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(34) :: acd10
      complex(ki) :: brack
      acd10(1)=k2(iv1)
      acd10(2)=k2(iv2)
      acd10(3)=abb10(10)
      acd10(4)=spval3k2(iv2)
      acd10(5)=abb10(30)
      acd10(6)=spval3k2(iv1)
      acd10(7)=spvak1k2(iv1)
      acd10(8)=spvak2k1(iv2)
      acd10(9)=abb10(9)
      acd10(10)=spval3k1(iv2)
      acd10(11)=abb10(23)
      acd10(12)=spvak1k2(iv2)
      acd10(13)=spvak2k1(iv1)
      acd10(14)=spval3k1(iv1)
      acd10(15)=spvak1l3(iv1)
      acd10(16)=spval5k1(iv2)
      acd10(17)=abb10(24)
      acd10(18)=spvak1l3(iv2)
      acd10(19)=spval5k1(iv1)
      acd10(20)=spvak1l4(iv2)
      acd10(21)=abb10(19)
      acd10(22)=spvak1l4(iv1)
      acd10(23)=spvak2l3(iv1)
      acd10(24)=spval5k2(iv2)
      acd10(25)=spvak2l3(iv2)
      acd10(26)=spval5k2(iv1)
      acd10(27)=spvak2l4(iv2)
      acd10(28)=spvak2l4(iv1)
      acd10(29)=-acd10(15)*acd10(16)
      acd10(30)=-acd10(18)*acd10(19)
      acd10(31)=acd10(23)*acd10(24)
      acd10(32)=acd10(25)*acd10(26)
      acd10(29)=acd10(32)+acd10(31)+acd10(30)+acd10(29)
      acd10(29)=acd10(17)*acd10(29)
      acd10(30)=-acd10(20)*acd10(19)
      acd10(31)=-acd10(22)*acd10(16)
      acd10(32)=acd10(27)*acd10(26)
      acd10(33)=acd10(28)*acd10(24)
      acd10(30)=acd10(33)+acd10(32)+acd10(31)+acd10(30)
      acd10(30)=acd10(21)*acd10(30)
      acd10(31)=-acd10(4)*acd10(1)
      acd10(32)=-acd10(6)*acd10(2)
      acd10(31)=acd10(32)+acd10(31)
      acd10(31)=acd10(5)*acd10(31)
      acd10(32)=acd10(8)*acd10(9)
      acd10(33)=acd10(10)*acd10(11)
      acd10(32)=acd10(33)+acd10(32)
      acd10(32)=acd10(7)*acd10(32)
      acd10(33)=acd10(13)*acd10(9)
      acd10(34)=acd10(14)*acd10(11)
      acd10(33)=acd10(34)+acd10(33)
      acd10(33)=acd10(12)*acd10(33)
      acd10(34)=acd10(3)*acd10(2)*acd10(1)
      brack=acd10(29)+acd10(30)+acd10(31)+acd10(32)+acd10(33)+2.0_ki*acd10(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd10h4_qp
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
end module     p2_gg_httbar_d10h4l1d_qp
