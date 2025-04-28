module     p2_gg_httbar_d11h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d11h4l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd11h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc11(41)
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspk1
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l4
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspk1 = dotproduct(Q,k1)
      Qspk2 = dotproduct(Q,k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      acc11(1)=abb11(9)
      acc11(2)=abb11(10)
      acc11(3)=abb11(11)
      acc11(4)=abb11(12)
      acc11(5)=abb11(13)
      acc11(6)=abb11(14)
      acc11(7)=abb11(15)
      acc11(8)=abb11(16)
      acc11(9)=abb11(17)
      acc11(10)=abb11(18)
      acc11(11)=abb11(19)
      acc11(12)=abb11(20)
      acc11(13)=abb11(21)
      acc11(14)=abb11(22)
      acc11(15)=abb11(23)
      acc11(16)=abb11(24)
      acc11(17)=abb11(25)
      acc11(18)=abb11(27)
      acc11(19)=abb11(29)
      acc11(20)=abb11(31)
      acc11(21)=abb11(32)
      acc11(22)=abb11(33)
      acc11(23)=abb11(34)
      acc11(24)=Qspval3k2*acc11(21)
      acc11(25)=Qspval5l3*acc11(11)
      acc11(26)=Qspval5l4*acc11(8)
      acc11(24)=acc11(26)+acc11(24)-acc11(25)
      acc11(25)=acc11(3)-acc11(24)
      acc11(25)=Qspk1*acc11(25)
      acc11(26)=Qspk2-Qspk1
      acc11(26)=acc11(23)*acc11(26)
      acc11(24)=acc11(1)+acc11(26)+acc11(24)
      acc11(24)=Qspk2*acc11(24)
      acc11(26)=acc11(22)*Qspval3l5
      acc11(27)=acc11(20)*Qspvak2l5
      acc11(28)=acc11(18)*Qspvak1l4
      acc11(29)=acc11(17)*Qspval5k1
      acc11(30)=acc11(16)*Qspvak1k2
      acc11(31)=acc11(15)*Qspvak2l3
      acc11(32)=acc11(14)*Qspval5k2
      acc11(33)=acc11(10)*Qspvak1l3
      acc11(34)=acc11(9)*Qspl5
      acc11(35)=acc11(7)*QspQ
      acc11(36)=acc11(5)*Qspval3k1
      acc11(37)=acc11(4)*Qspvak2k1
      acc11(38)=acc11(2)*Qspvak2l4
      acc11(39)=Qspval3k2*acc11(19)
      acc11(40)=Qspval5l3*acc11(13)
      acc11(41)=Qspval5l4*acc11(12)
      brack=acc11(6)+acc11(24)+acc11(25)+acc11(26)+acc11(27)+acc11(28)+acc11(29&
      &)+acc11(30)+acc11(31)+acc11(32)+acc11(33)+acc11(34)+acc11(35)+acc11(36)+&
      &acc11(37)+acc11(38)+acc11(39)+acc11(40)+acc11(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d11h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd11h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d11
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d11 = 0.0_ki
      d11 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d11, ki), aimag(d11), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d11h4l1_qp
