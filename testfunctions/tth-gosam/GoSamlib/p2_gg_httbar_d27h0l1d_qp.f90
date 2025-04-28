module     p2_gg_httbar_d27h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d27h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd27h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(48) :: acd27
      complex(ki) :: brack
      acd27(1)=dotproduct(qshift,spvak1e2)
      acd27(2)=dotproduct(qshift,spvae2k2)
      acd27(3)=abb27(20)
      acd27(4)=dotproduct(qshift,spvae2l3)
      acd27(5)=abb27(17)
      acd27(6)=abb27(16)
      acd27(7)=dotproduct(qshift,spvak2e2)
      acd27(8)=abb27(19)
      acd27(9)=dotproduct(qshift,spval4e2)
      acd27(10)=abb27(24)
      acd27(11)=dotproduct(qshift,spvae1e2)
      acd27(12)=abb27(15)
      acd27(13)=abb27(12)
      acd27(14)=abb27(28)
      acd27(15)=abb27(43)
      acd27(16)=abb27(34)
      acd27(17)=abb27(41)
      acd27(18)=dotproduct(qshift,spvae2k1)
      acd27(19)=dotproduct(qshift,spval3e2)
      acd27(20)=abb27(49)
      acd27(21)=dotproduct(qshift,spval5e2)
      acd27(22)=abb27(63)
      acd27(23)=abb27(13)
      acd27(24)=dotproduct(qshift,spvae2l4)
      acd27(25)=abb27(36)
      acd27(26)=dotproduct(qshift,spvae2e1)
      acd27(27)=abb27(50)
      acd27(28)=abb27(30)
      acd27(29)=abb27(27)
      acd27(30)=abb27(46)
      acd27(31)=abb27(11)
      acd27(32)=abb27(26)
      acd27(33)=abb27(31)
      acd27(34)=abb27(25)
      acd27(35)=abb27(18)
      acd27(36)=abb27(14)
      acd27(37)=abb27(10)
      acd27(38)=acd27(3)*acd27(1)
      acd27(39)=acd27(8)*acd27(7)
      acd27(40)=acd27(10)*acd27(9)
      acd27(41)=acd27(12)*acd27(11)
      acd27(38)=-acd27(13)+acd27(41)+acd27(40)+acd27(39)+acd27(38)
      acd27(38)=acd27(2)*acd27(38)
      acd27(39)=acd27(5)*acd27(1)
      acd27(40)=acd27(14)*acd27(7)
      acd27(41)=acd27(15)*acd27(9)
      acd27(42)=acd27(16)*acd27(11)
      acd27(39)=-acd27(17)+acd27(42)+acd27(41)+acd27(40)+acd27(39)
      acd27(39)=acd27(4)*acd27(39)
      acd27(40)=acd27(20)*acd27(18)
      acd27(41)=acd27(25)*acd27(24)
      acd27(42)=-acd27(27)*acd27(26)
      acd27(40)=-acd27(28)+acd27(42)+acd27(41)+acd27(40)
      acd27(40)=acd27(19)*acd27(40)
      acd27(41)=acd27(22)*acd27(18)
      acd27(42)=acd27(29)*acd27(24)
      acd27(43)=-acd27(30)*acd27(26)
      acd27(41)=-acd27(31)+acd27(43)+acd27(42)+acd27(41)
      acd27(41)=acd27(21)*acd27(41)
      acd27(42)=-acd27(6)*acd27(1)
      acd27(43)=-acd27(23)*acd27(18)
      acd27(44)=-acd27(32)*acd27(7)
      acd27(45)=-acd27(33)*acd27(9)
      acd27(46)=-acd27(34)*acd27(11)
      acd27(47)=-acd27(35)*acd27(24)
      acd27(48)=-acd27(36)*acd27(26)
      brack=acd27(37)+acd27(38)+acd27(39)+acd27(40)+acd27(41)+acd27(42)+acd27(4&
      &3)+acd27(44)+acd27(45)+acd27(46)+acd27(47)+acd27(48)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd27h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(62) :: acd27
      complex(ki) :: brack
      acd27(1)=spvak1e2(iv1)
      acd27(2)=dotproduct(qshift,spvae2k2)
      acd27(3)=abb27(20)
      acd27(4)=dotproduct(qshift,spvae2l3)
      acd27(5)=abb27(17)
      acd27(6)=abb27(16)
      acd27(7)=spvae2k2(iv1)
      acd27(8)=dotproduct(qshift,spvak1e2)
      acd27(9)=dotproduct(qshift,spvak2e2)
      acd27(10)=abb27(19)
      acd27(11)=dotproduct(qshift,spval4e2)
      acd27(12)=abb27(24)
      acd27(13)=dotproduct(qshift,spvae1e2)
      acd27(14)=abb27(15)
      acd27(15)=abb27(12)
      acd27(16)=spvae2l3(iv1)
      acd27(17)=abb27(28)
      acd27(18)=abb27(43)
      acd27(19)=abb27(34)
      acd27(20)=abb27(41)
      acd27(21)=spvae2k1(iv1)
      acd27(22)=dotproduct(qshift,spval3e2)
      acd27(23)=abb27(49)
      acd27(24)=dotproduct(qshift,spval5e2)
      acd27(25)=abb27(63)
      acd27(26)=abb27(13)
      acd27(27)=spval3e2(iv1)
      acd27(28)=dotproduct(qshift,spvae2k1)
      acd27(29)=dotproduct(qshift,spvae2l4)
      acd27(30)=abb27(36)
      acd27(31)=dotproduct(qshift,spvae2e1)
      acd27(32)=abb27(50)
      acd27(33)=abb27(30)
      acd27(34)=spval5e2(iv1)
      acd27(35)=abb27(27)
      acd27(36)=abb27(46)
      acd27(37)=abb27(11)
      acd27(38)=spvak2e2(iv1)
      acd27(39)=abb27(26)
      acd27(40)=spval4e2(iv1)
      acd27(41)=abb27(31)
      acd27(42)=spvae1e2(iv1)
      acd27(43)=abb27(25)
      acd27(44)=spvae2l4(iv1)
      acd27(45)=abb27(18)
      acd27(46)=spvae2e1(iv1)
      acd27(47)=abb27(14)
      acd27(48)=acd27(42)*acd27(19)
      acd27(49)=acd27(40)*acd27(18)
      acd27(50)=acd27(38)*acd27(17)
      acd27(51)=acd27(1)*acd27(5)
      acd27(48)=acd27(51)+acd27(50)+acd27(48)+acd27(49)
      acd27(48)=acd27(4)*acd27(48)
      acd27(49)=acd27(42)*acd27(14)
      acd27(50)=acd27(40)*acd27(12)
      acd27(51)=acd27(38)*acd27(10)
      acd27(52)=acd27(1)*acd27(3)
      acd27(49)=acd27(52)+acd27(51)+acd27(49)+acd27(50)
      acd27(49)=acd27(2)*acd27(49)
      acd27(50)=acd27(13)*acd27(19)
      acd27(51)=acd27(11)*acd27(18)
      acd27(52)=acd27(9)*acd27(17)
      acd27(53)=acd27(5)*acd27(8)
      acd27(50)=acd27(53)+acd27(52)+acd27(51)-acd27(20)+acd27(50)
      acd27(50)=acd27(16)*acd27(50)
      acd27(51)=acd27(13)*acd27(14)
      acd27(52)=acd27(11)*acd27(12)
      acd27(53)=acd27(9)*acd27(10)
      acd27(54)=acd27(3)*acd27(8)
      acd27(51)=acd27(54)+acd27(53)+acd27(52)-acd27(15)+acd27(51)
      acd27(51)=acd27(7)*acd27(51)
      acd27(52)=-acd27(31)*acd27(36)
      acd27(53)=acd27(29)*acd27(35)
      acd27(54)=acd27(25)*acd27(28)
      acd27(52)=acd27(54)+acd27(53)-acd27(37)+acd27(52)
      acd27(52)=acd27(34)*acd27(52)
      acd27(53)=-acd27(31)*acd27(32)
      acd27(54)=acd27(29)*acd27(30)
      acd27(55)=acd27(23)*acd27(28)
      acd27(53)=acd27(55)+acd27(54)-acd27(33)+acd27(53)
      acd27(53)=acd27(27)*acd27(53)
      acd27(54)=-acd27(46)*acd27(36)
      acd27(55)=acd27(44)*acd27(35)
      acd27(54)=acd27(54)+acd27(55)
      acd27(54)=acd27(24)*acd27(54)
      acd27(55)=-acd27(46)*acd27(32)
      acd27(56)=acd27(44)*acd27(30)
      acd27(55)=acd27(55)+acd27(56)
      acd27(55)=acd27(22)*acd27(55)
      acd27(56)=acd27(24)*acd27(25)
      acd27(57)=acd27(22)*acd27(23)
      acd27(56)=acd27(57)-acd27(26)+acd27(56)
      acd27(56)=acd27(21)*acd27(56)
      acd27(57)=-acd27(46)*acd27(47)
      acd27(58)=-acd27(44)*acd27(45)
      acd27(59)=-acd27(42)*acd27(43)
      acd27(60)=-acd27(40)*acd27(41)
      acd27(61)=-acd27(38)*acd27(39)
      acd27(62)=-acd27(1)*acd27(6)
      brack=acd27(48)+acd27(49)+acd27(50)+acd27(51)+acd27(52)+acd27(53)+acd27(5&
      &4)+acd27(55)+acd27(56)+acd27(57)+acd27(58)+acd27(59)+acd27(60)+acd27(61)&
      &+acd27(62)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd27h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(46) :: acd27
      complex(ki) :: brack
      acd27(1)=spvak1e2(iv1)
      acd27(2)=spvae2k2(iv2)
      acd27(3)=abb27(20)
      acd27(4)=spvae2l3(iv2)
      acd27(5)=abb27(17)
      acd27(6)=spvak1e2(iv2)
      acd27(7)=spvae2k2(iv1)
      acd27(8)=spvae2l3(iv1)
      acd27(9)=spvak2e2(iv2)
      acd27(10)=abb27(19)
      acd27(11)=spval4e2(iv2)
      acd27(12)=abb27(24)
      acd27(13)=spvae1e2(iv2)
      acd27(14)=abb27(15)
      acd27(15)=spvak2e2(iv1)
      acd27(16)=spval4e2(iv1)
      acd27(17)=spvae1e2(iv1)
      acd27(18)=abb27(28)
      acd27(19)=abb27(43)
      acd27(20)=abb27(34)
      acd27(21)=spvae2k1(iv1)
      acd27(22)=spval3e2(iv2)
      acd27(23)=abb27(49)
      acd27(24)=spval5e2(iv2)
      acd27(25)=abb27(63)
      acd27(26)=spvae2k1(iv2)
      acd27(27)=spval3e2(iv1)
      acd27(28)=spval5e2(iv1)
      acd27(29)=spvae2l4(iv2)
      acd27(30)=abb27(36)
      acd27(31)=spvae2e1(iv2)
      acd27(32)=abb27(50)
      acd27(33)=spvae2l4(iv1)
      acd27(34)=spvae2e1(iv1)
      acd27(35)=abb27(27)
      acd27(36)=abb27(46)
      acd27(37)=acd27(13)*acd27(20)
      acd27(38)=acd27(11)*acd27(19)
      acd27(39)=acd27(9)*acd27(18)
      acd27(40)=acd27(5)*acd27(6)
      acd27(37)=acd27(40)+acd27(39)+acd27(37)+acd27(38)
      acd27(37)=acd27(8)*acd27(37)
      acd27(38)=acd27(13)*acd27(14)
      acd27(39)=acd27(11)*acd27(12)
      acd27(40)=acd27(9)*acd27(10)
      acd27(41)=acd27(3)*acd27(6)
      acd27(38)=acd27(41)+acd27(40)+acd27(38)+acd27(39)
      acd27(38)=acd27(7)*acd27(38)
      acd27(39)=acd27(17)*acd27(20)
      acd27(40)=acd27(16)*acd27(19)
      acd27(41)=acd27(15)*acd27(18)
      acd27(42)=acd27(1)*acd27(5)
      acd27(39)=acd27(42)+acd27(41)+acd27(39)+acd27(40)
      acd27(39)=acd27(4)*acd27(39)
      acd27(40)=acd27(14)*acd27(17)
      acd27(41)=acd27(12)*acd27(16)
      acd27(42)=acd27(10)*acd27(15)
      acd27(43)=acd27(1)*acd27(3)
      acd27(40)=acd27(43)+acd27(42)+acd27(40)+acd27(41)
      acd27(40)=acd27(2)*acd27(40)
      acd27(41)=-acd27(31)*acd27(36)
      acd27(42)=acd27(29)*acd27(35)
      acd27(43)=acd27(25)*acd27(26)
      acd27(41)=acd27(43)+acd27(41)+acd27(42)
      acd27(41)=acd27(28)*acd27(41)
      acd27(42)=-acd27(31)*acd27(32)
      acd27(43)=acd27(29)*acd27(30)
      acd27(44)=acd27(23)*acd27(26)
      acd27(42)=acd27(44)+acd27(42)+acd27(43)
      acd27(42)=acd27(27)*acd27(42)
      acd27(43)=-acd27(34)*acd27(36)
      acd27(44)=acd27(33)*acd27(35)
      acd27(45)=acd27(21)*acd27(25)
      acd27(43)=acd27(45)+acd27(43)+acd27(44)
      acd27(43)=acd27(24)*acd27(43)
      acd27(44)=-acd27(32)*acd27(34)
      acd27(45)=acd27(30)*acd27(33)
      acd27(46)=acd27(21)*acd27(23)
      acd27(44)=acd27(46)+acd27(44)+acd27(45)
      acd27(44)=acd27(22)*acd27(44)
      brack=acd27(37)+acd27(38)+acd27(39)+acd27(40)+acd27(41)+acd27(42)+acd27(4&
      &3)+acd27(44)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd27h0_qp
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
      qshift = -k2+k3+k5
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
end module     p2_gg_httbar_d27h0l1d_qp
