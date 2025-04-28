module     p2_gg_httbar_d177h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d177h12l1d_qp.f90
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
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd177
      complex(ki) :: brack
      acd177(1)=dotproduct(qshift,spvak1e1)
      acd177(2)=dotproduct(qshift,spvae1l4)
      acd177(3)=abb177(14)
      acd177(4)=abb177(24)
      acd177(5)=dotproduct(qshift,spvak2e1)
      acd177(6)=abb177(28)
      acd177(7)=dotproduct(qshift,spval4e1)
      acd177(8)=abb177(12)
      acd177(9)=dotproduct(qshift,spval5e1)
      acd177(10)=abb177(20)
      acd177(11)=dotproduct(qshift,spvae2e1)
      acd177(12)=abb177(40)
      acd177(13)=abb177(15)
      acd177(14)=dotproduct(qshift,spvae1k1)
      acd177(15)=abb177(44)
      acd177(16)=abb177(22)
      acd177(17)=dotproduct(qshift,spvae1k2)
      acd177(18)=abb177(19)
      acd177(19)=dotproduct(qshift,spvae1l5)
      acd177(20)=abb177(27)
      acd177(21)=dotproduct(qshift,spvae1e2)
      acd177(22)=abb177(26)
      acd177(23)=abb177(13)
      acd177(24)=abb177(23)
      acd177(25)=abb177(59)
      acd177(26)=abb177(52)
      acd177(27)=abb177(46)
      acd177(28)=abb177(65)
      acd177(29)=abb177(21)
      acd177(30)=abb177(25)
      acd177(31)=acd177(6)*acd177(2)
      acd177(32)=acd177(15)*acd177(14)
      acd177(33)=acd177(18)*acd177(17)
      acd177(34)=acd177(20)*acd177(19)
      acd177(35)=acd177(22)*acd177(21)
      acd177(31)=-acd177(23)+acd177(35)+acd177(34)+acd177(33)+acd177(32)+acd177&
      &(31)
      acd177(31)=acd177(5)*acd177(31)
      acd177(32)=acd177(3)*acd177(1)
      acd177(33)=acd177(8)*acd177(7)
      acd177(34)=-acd177(10)*acd177(9)
      acd177(35)=-acd177(12)*acd177(11)
      acd177(32)=-acd177(13)+acd177(35)+acd177(34)+acd177(33)+acd177(32)
      acd177(32)=acd177(2)*acd177(32)
      acd177(33)=-acd177(4)*acd177(1)
      acd177(34)=-acd177(16)*acd177(14)
      acd177(35)=-acd177(24)*acd177(17)
      acd177(36)=-acd177(25)*acd177(19)
      acd177(37)=-acd177(26)*acd177(21)
      acd177(38)=-acd177(27)*acd177(7)
      acd177(39)=-acd177(28)*acd177(9)
      acd177(40)=-acd177(29)*acd177(11)
      brack=acd177(30)+acd177(31)+acd177(32)+acd177(33)+acd177(34)+acd177(35)+a&
      &cd177(36)+acd177(37)+acd177(38)+acd177(39)+acd177(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(51) :: acd177
      complex(ki) :: brack
      acd177(1)=spvak1e1(iv1)
      acd177(2)=dotproduct(qshift,spvae1l4)
      acd177(3)=abb177(14)
      acd177(4)=abb177(24)
      acd177(5)=spvae1l4(iv1)
      acd177(6)=dotproduct(qshift,spvak1e1)
      acd177(7)=dotproduct(qshift,spvak2e1)
      acd177(8)=abb177(28)
      acd177(9)=dotproduct(qshift,spval4e1)
      acd177(10)=abb177(12)
      acd177(11)=dotproduct(qshift,spval5e1)
      acd177(12)=abb177(20)
      acd177(13)=dotproduct(qshift,spvae2e1)
      acd177(14)=abb177(40)
      acd177(15)=abb177(15)
      acd177(16)=spvae1k1(iv1)
      acd177(17)=abb177(44)
      acd177(18)=abb177(22)
      acd177(19)=spvak2e1(iv1)
      acd177(20)=dotproduct(qshift,spvae1k1)
      acd177(21)=dotproduct(qshift,spvae1k2)
      acd177(22)=abb177(19)
      acd177(23)=dotproduct(qshift,spvae1l5)
      acd177(24)=abb177(27)
      acd177(25)=dotproduct(qshift,spvae1e2)
      acd177(26)=abb177(26)
      acd177(27)=abb177(13)
      acd177(28)=spvae1k2(iv1)
      acd177(29)=abb177(23)
      acd177(30)=spvae1l5(iv1)
      acd177(31)=abb177(59)
      acd177(32)=spvae1e2(iv1)
      acd177(33)=abb177(52)
      acd177(34)=spval4e1(iv1)
      acd177(35)=abb177(46)
      acd177(36)=spval5e1(iv1)
      acd177(37)=abb177(65)
      acd177(38)=spvae2e1(iv1)
      acd177(39)=abb177(21)
      acd177(40)=acd177(26)*acd177(25)
      acd177(41)=acd177(24)*acd177(23)
      acd177(42)=acd177(22)*acd177(21)
      acd177(43)=acd177(17)*acd177(20)
      acd177(44)=acd177(2)*acd177(8)
      acd177(40)=acd177(44)+acd177(43)+acd177(42)+acd177(41)-acd177(27)+acd177(&
      &40)
      acd177(40)=acd177(19)*acd177(40)
      acd177(41)=-acd177(14)*acd177(13)
      acd177(42)=-acd177(12)*acd177(11)
      acd177(43)=acd177(10)*acd177(9)
      acd177(44)=acd177(3)*acd177(6)
      acd177(45)=acd177(7)*acd177(8)
      acd177(41)=acd177(45)+acd177(44)+acd177(43)+acd177(42)-acd177(15)+acd177(&
      &41)
      acd177(41)=acd177(5)*acd177(41)
      acd177(42)=acd177(26)*acd177(32)
      acd177(43)=acd177(24)*acd177(30)
      acd177(44)=acd177(22)*acd177(28)
      acd177(45)=acd177(16)*acd177(17)
      acd177(42)=acd177(45)+acd177(44)+acd177(42)+acd177(43)
      acd177(42)=acd177(7)*acd177(42)
      acd177(43)=-acd177(14)*acd177(38)
      acd177(44)=-acd177(12)*acd177(36)
      acd177(45)=acd177(10)*acd177(34)
      acd177(46)=acd177(1)*acd177(3)
      acd177(43)=acd177(46)+acd177(45)+acd177(43)+acd177(44)
      acd177(43)=acd177(2)*acd177(43)
      acd177(44)=-acd177(38)*acd177(39)
      acd177(45)=-acd177(36)*acd177(37)
      acd177(46)=-acd177(34)*acd177(35)
      acd177(47)=-acd177(32)*acd177(33)
      acd177(48)=-acd177(30)*acd177(31)
      acd177(49)=-acd177(28)*acd177(29)
      acd177(50)=-acd177(16)*acd177(18)
      acd177(51)=-acd177(1)*acd177(4)
      brack=acd177(40)+acd177(41)+acd177(42)+acd177(43)+acd177(44)+acd177(45)+a&
      &cd177(46)+acd177(47)+acd177(48)+acd177(49)+acd177(50)+acd177(51)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd177h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(36) :: acd177
      complex(ki) :: brack
      acd177(1)=spvak1e1(iv1)
      acd177(2)=spvae1l4(iv2)
      acd177(3)=abb177(14)
      acd177(4)=spvak1e1(iv2)
      acd177(5)=spvae1l4(iv1)
      acd177(6)=spvak2e1(iv2)
      acd177(7)=abb177(28)
      acd177(8)=spval4e1(iv2)
      acd177(9)=abb177(12)
      acd177(10)=spval5e1(iv2)
      acd177(11)=abb177(20)
      acd177(12)=spvae2e1(iv2)
      acd177(13)=abb177(40)
      acd177(14)=spvak2e1(iv1)
      acd177(15)=spval4e1(iv1)
      acd177(16)=spval5e1(iv1)
      acd177(17)=spvae2e1(iv1)
      acd177(18)=spvae1k1(iv1)
      acd177(19)=abb177(44)
      acd177(20)=spvae1k1(iv2)
      acd177(21)=spvae1k2(iv2)
      acd177(22)=abb177(19)
      acd177(23)=spvae1l5(iv2)
      acd177(24)=abb177(27)
      acd177(25)=spvae1e2(iv2)
      acd177(26)=abb177(26)
      acd177(27)=spvae1k2(iv1)
      acd177(28)=spvae1l5(iv1)
      acd177(29)=spvae1e2(iv1)
      acd177(30)=-acd177(13)*acd177(12)
      acd177(31)=-acd177(11)*acd177(10)
      acd177(32)=acd177(9)*acd177(8)
      acd177(33)=acd177(3)*acd177(4)
      acd177(34)=acd177(6)*acd177(7)
      acd177(30)=acd177(34)+acd177(33)+acd177(32)+acd177(30)+acd177(31)
      acd177(30)=acd177(5)*acd177(30)
      acd177(31)=-acd177(13)*acd177(17)
      acd177(32)=-acd177(11)*acd177(16)
      acd177(33)=acd177(9)*acd177(15)
      acd177(34)=acd177(3)*acd177(1)
      acd177(35)=acd177(14)*acd177(7)
      acd177(31)=acd177(35)+acd177(34)+acd177(33)+acd177(31)+acd177(32)
      acd177(31)=acd177(2)*acd177(31)
      acd177(32)=acd177(26)*acd177(25)
      acd177(33)=acd177(24)*acd177(23)
      acd177(34)=acd177(22)*acd177(21)
      acd177(35)=acd177(19)*acd177(20)
      acd177(32)=acd177(35)+acd177(34)+acd177(32)+acd177(33)
      acd177(32)=acd177(14)*acd177(32)
      acd177(33)=acd177(26)*acd177(29)
      acd177(34)=acd177(24)*acd177(28)
      acd177(35)=acd177(22)*acd177(27)
      acd177(36)=acd177(19)*acd177(18)
      acd177(33)=acd177(36)+acd177(35)+acd177(33)+acd177(34)
      acd177(33)=acd177(6)*acd177(33)
      brack=acd177(30)+acd177(31)+acd177(32)+acd177(33)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd177h12_qp
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
      qshift = -k4
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
end module     p2_gg_httbar_d177h12l1d_qp
